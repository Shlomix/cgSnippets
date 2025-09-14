# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Flash Attention Layer (BSH). Prefill-1: FA. Prefill-2+: FA (dense KV via block gather). Decode: PagedAttention."""
__all__ = ['FlashAttention']

import os
from mindspore import ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """
    BSH FlashAttention + PagedAttention bridge.

    Behavior
      • First prefill (no prior ctx): FlashAttention (BSH).
      • Second+ prefill (has prior ctx): FlashAttention (BSH) over full prefix by building dense KV from paged cache
        using **blockwise gather** (batch=1 PoC). If constraints not met, fallback to PagedAttention (unless forced).
      • Decode: PagedAttention.

    Env toggles
      MF_DEBUG_ATTENTION=1        -> print key steps.
      MF_FORCE_FA_CHUNK=1         -> (default=1) attempt FA on 2nd+ prefill (else PA).
      MF_MANIFEST_FA_MISMATCH=1   -> force FA even if not x16 or q_len!=kv_len (to demonstrate kernel failure).
    """

    def __init__(self,
                 head_num,
                 head_dim=None,
                 kv_head_num=None,
                 keep_prob=1.0,
                 scale_value=1.0,
                 pre_tokens=2147483647,
                 next_tokens=2147483647,
                 sparse_mode=0,
                 input_layout="BSH",
                 pa_kv_head_num=None,
                 pa_mla_v_dim=0):
        super().__init__()
        if input_layout != "BSH":
            raise ValueError("FlashAttention wrapper expects input_layout='BSH' in this branch.")

        self.head_num = int(head_num)
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.kv_head_num = int(kv_head_num) if kv_head_num is not None else None
        self.sparse_mode = sparse_mode
        self.is_prefill = True
        self.use_multi_latent_attention: bool = pa_mla_v_dim > 0

        # Env toggles
        self.debug = os.getenv("MF_DEBUG_ATTENTION", "0") == "1"
        self.force_fa_chunk = os.getenv("MF_FORCE_FA_CHUNK", "1") == "1"
        self.manifest_mismatch = os.getenv("MF_MANIFEST_FA_MISMATCH", "0") == "1"

        # Ops
        self._Cast = ops.Cast()
        self._Reshape = ops.Reshape()
        self._Print = ops.Print()
        self._Range = ops.Range()
        self._ExpandDims = ops.ExpandDims()
        self._Add = ops.Add()
        self._Mul = ops.Mul()
        self._Mod = ops.Mod()
        self._ReduceSum = ops.ReduceSum(keep_dims=False)
        self._ReduceMax = ops.ReduceMax(keep_dims=False)
        self._Gather = ops.Gather()
        self._FloorDiv = ops.FloorDiv()
        self._Minimum = ops.Minimum()
        self._Maximum = ops.Maximum()
        self._Concat = ops.Concat(axis=1)
        self._BroadcastTo = ops.BroadcastTo

        # Kernels
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout="BSH",
            sparse_mode=self.sparse_mode
        )
        self.paged_attention = ops.auto_generate.PagedAttention(
            head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim
        )
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

    # ---------------- debug helper ----------------
    def _dbg(self, *args):
        if self.debug:
            self._Print(*args)

    # ---------------- small helpers ----------------
    def _all_x16(self, x_i32):
        rem = self._Mod(x_i32, ops.scalar_to_tensor(16, mstype.int32))
        s = self._ReduceSum(rem, 0)
        return s == ops.scalar_to_tensor(0, mstype.int32)

    def _repeat_kv_to_q_heads(self, dense_kv):
        """[T_kv, kv_heads*head_dim] -> [T_kv, head_num*head_dim] if kv_heads < head_num."""
        if self.kv_head_num is None or self.head_dim is None:
            return dense_kv
        if self.kv_head_num == self.head_num:
            return dense_kv
        rep = self.head_num // self.kv_head_num
        x = self._Reshape(dense_kv, (-1, self.kv_head_num, self.head_dim))
        x = self._Concat(tuple([x] * rep))  # concat along head axis (axis=1)
        x = self._Reshape(x, (-1, self.head_num * self.head_dim))
        return x

    def _sanitize_fa_inputs(self, q_bsh, k_bsh, attn_mask, aql, akl):
        """For FA: cast attn_mask to q.dtype and ensure shapes [B,Sq,Sk]. Lengths int32."""
        B, Sq, _ = q_bsh.shape
        Sk = int(k_bsh.shape[1])
        qdtype = q_bsh.dtype

        # attn_mask -> [B,Sq,Sk]
        if attn_mask is None:
            attn_mask = ops.zeros((B, Sq, Sk), qdtype)
        else:
            am = attn_mask
            if len(am.shape) == 2:
                am = self._ExpandDims(am, 0)           # [1,Sq,Sk]
            if am.shape[0] == 1 and B > 1:
                am = self._BroadcastTo((B, am.shape[1], am.shape[2]))(am)
            if am.dtype != qdtype:
                am = self._Cast(am, qdtype)
            attn_mask = am

        if aql is None:
            aql = ops.fill(mstype.int32, (B,), Sq)
        elif aql.dtype != mstype.int32:
            aql = self._Cast(aql, mstype.int32)

        if akl is None:
            akl = ops.fill(mstype.int32, (B,), Sk)
        elif akl.dtype != mstype.int32:
            akl = self._Cast(akl, mstype.int32)

        return attn_mask, aql, akl

    def _dense_kv_from_cache_b1(self, key_cache, value_cache, block_tables, kv_end_i32):
        """
        Build dense KV contiguously from paged cache for B=1 using **blockwise gather** (no token index vector).
        key_cache: [num_blocks, block_size, kv_heads*head_dim] or [num_blocks, block_size, kv_heads, head_dim]
        block_tables: [1, num_blocks] int32 with physical block ids (unused tail may be -1).
        kv_end_i32: scalar int32 (prefix length to include, <= num_blocks*block_size).
        Returns: K_dense, V_dense TH: [T_kv, H_kv], and block_size (int).
        """
        bt = block_tables[0]                     # [num_blocks]
        shape = key_cache.shape
        if len(shape) == 4:
            nb, bs, kvh, hd = shape
            H_kv = kvh * hd
            kc_nbbsH = self._Reshape(key_cache, (nb, bs, H_kv))
            vc_nbbsH = self._Reshape(value_cache, (nb, bs, H_kv))
        elif len(shape) == 3:
            nb, bs, H_kv = shape
            kc_nbbsH = key_cache
            vc_nbbsH = value_cache
        else:
            nb, bs, H_kv = int(shape[0]), int(shape[1]), int(shape[-1])
            kc_nbbsH = self._Reshape(key_cache, (nb, bs, H_kv))
            vc_nbbsH = self._Reshape(value_cache, (nb, bs, H_kv))

        # used_blocks = min(nb, ceil(kv_end/bs))
        bs_t = ops.scalar_to_tensor(bs, mstype.int32)
        nb_t = ops.scalar_to_tensor(nb, mstype.int32)
        used_blocks = self._FloorDiv(kv_end_i32 + (bs_t - ops.scalar_to_tensor(1, mstype.int32)), bs_t)
        used_blocks = self._Minimum(used_blocks, nb_t)

        # range [0..used_blocks)
        rng = self._Range(ops.scalar_to_tensor(0, mstype.int32),
                          used_blocks,
                          ops.scalar_to_tensor(1, mstype.int32))

        # physical ids for the first used blocks, clamp to [0, nb-1] to avoid -1/OOB
        zero = ops.scalar_to_tensor(0, mstype.int32)
        nbm1 = ops.scalar_to_tensor(nb - 1, mstype.int32)
        ids = self._Gather(bt, rng, 0)                  # [used_blocks]
        ids = self._Maximum(ids, zero)
        ids = self._Minimum(ids, nbm1)

        # gather **blocks** (safer than building per-token indices)
        kc_used = self._Gather(kc_nbbsH, ids, 0)        # [u, bs, H_kv]
        vc_used = self._Gather(vc_nbbsH, ids, 0)        # [u, bs, H_kv]

        # flatten to [u*bs, H_kv], then **contiguous** slice first kv_end tokens
        kv_flat = self._Reshape(kc_used, (-1, H_kv))    # [u*bs, H_kv]
        vv_flat = self._Reshape(vc_used, (-1, H_kv))    # [u*bs, H_kv]

        # indices 0..kv_end-1
        idx0 = self._Range(zero, kv_end_i32, ops.scalar_to_tensor(1, mstype.int32))
        k_dense = self._Gather(kv_flat, idx0, 0)        # [kv_end, H_kv]
        v_dense = self._Gather(vv_flat, idx0, 0)
        return k_dense, v_dense, bs

    def _run_paged(self, query, key_cache, value_cache, block_tables, batch_valid_length, q_seq_lens, attn_mask):
        want = (key_cache.dtype if key_cache is not None else query.dtype)
        if attn_mask is not None and attn_mask.dtype != want:
            attn_mask = self._Cast(attn_mask, want)
        if self.use_multi_latent_attention:
            return self.paged_attention(query, key_cache, key_cache,
                                        block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)
        return self.paged_attention(query, key_cache, value_cache,
                                    block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)

    # ---------------- forward ----------------
    def construct(self,
                  query,            # [B,Sq,H]
                  key,              # [B,Sq,H_kv]
                  value,            # [B,Sq,H_kv]
                  slot_mapping=None,
                  block_tables=None,
                  batch_valid_length=None,  # [B] context length BEFORE this chunk
                  context_lens_tensor=None,
                  q_seq_lens=None,          # [B]
                  actual_seq_qlen=None,     # [B]
                  actual_seq_kvlen=None,    # [B]
                  attn_mask=None,           # [B,Sq,Sk] or [Sq,Sk]
                  padding_mask=None,        # MUST be None for FA
                  prefix=None,
                  key_cache=None,
                  value_cache=None):
        """Forward process."""
        B, Sq, _ = query.shape
        caches_exist = (key_cache is not None) and (value_cache is not None)
        target_dtype = key_cache.dtype if caches_exist else query.dtype

        # align dtypes
        if query.dtype != target_dtype:
            query = self._Cast(query, target_dtype)
        if key is not None and key.dtype != target_dtype:
            key = self._Cast(key, target_dtype)
        if value is not None and value.dtype != target_dtype:
            value = self._Cast(value, target_dtype)

        # write current chunk into cache if available (needed later by PA/FA-2+)
        if caches_exist and (slot_mapping is not None) and (not self.use_multi_latent_attention):
            if slot_mapping.dtype != mstype.int32:
                slot_mapping = self._Cast(slot_mapping, mstype.int32)
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # ---------------- PREFILL ----------------
        if self.is_prefill:
            padding_mask = None  # Ascend FA requirement

            has_prior_ctx = (batch_valid_length is not None) and \
                            (int(self._ReduceMax(self._Cast(batch_valid_length, mstype.int32)).asnumpy()) > 0)

            # Prefill-1: FA on current chunk tensors
            if not has_prior_ctx:
                aql = actual_seq_qlen if actual_seq_qlen is not None else ops.fill(mstype.int32, (B,), Sq)
                akl = actual_seq_kvlen if actual_seq_kvlen is not None else ops.fill(mstype.int32, (B,), Sq)
                aql = self._Cast(aql, mstype.int32)
                akl = self._Cast(akl, mstype.int32)

                aligned = self._all_x16(aql) and self._all_x16(akl)
                if not aligned and not self.manifest_mismatch:
                    self._dbg("Prefill-1: not x16 -> PA fallback; aql=", aql, " akl=", akl)
                    return self._run_paged(query, key_cache, value_cache, block_tables,
                                           batch_valid_length, q_seq_lens, attn_mask)

                am, aql, akl = self._sanitize_fa_inputs(query, key, attn_mask, aql, akl)
                self._dbg("Prefill-1: FA; aql=", aql, " akl=", akl, " dtype=", query.dtype)
                _, _, _, ctx = self.flash_attention(query, key, value,
                                                    None, None,  # real_shift, alibi
                                                    None,        # padding_mask MUST be None
                                                    am,          # attn_mask
                                                    None,        # prefix
                                                    aql, akl)
                return ctx

            # Prefill-2+: FA over full prefix (batch=1 PoC) or fallback to PA
            if not self.force_fa_chunk:
                self._dbg("Prefill-2+: forced off -> PA")
                return self._run_paged(query, key_cache, value_cache, block_tables,
                                       batch_valid_length, q_seq_lens, attn_mask)

            if B != 1 and not self.manifest_mismatch:
                self._dbg("Prefill-2+: B!=1 -> PA fallback")
                return self._run_paged(query, key_cache, value_cache, block_tables,
                                       batch_valid_length, q_seq_lens, attn_mask)

            if q_seq_lens is None:
                q_seq_lens = ops.fill(mstype.int32, (B,), Sq)
            q_seq_lens = self._Cast(q_seq_lens, mstype.int32)
            bvl = self._Cast(batch_valid_length, mstype.int32)
            kv_end = self._Add(bvl, q_seq_lens)  # [1]

            aligned = self._all_x16(q_seq_lens) and self._all_x16(kv_end)
            if (not aligned) and (not self.manifest_mismatch):
                self._dbg("Prefill-2+: not x16 -> PA fallback; q=", q_seq_lens, " kv_end=", kv_end)
                return self._run_paged(query, key_cache, value_cache, block_tables,
                                       batch_valid_length, q_seq_lens, attn_mask)

            # safe dense KV gather from cache (B=1) using **block gather + contiguous slice**
            bt_i32 = self._Cast(block_tables, mstype.int32)
            k_dense, v_dense, block_size = self._dense_kv_from_cache_b1(
                key_cache, value_cache, bt_i32, kv_end[0]
            )  # [T_kv, H_kv]

            # head repeat if needed
            k_dense = self._repeat_kv_to_q_heads(k_dense)  # [T_kv, H]
            v_dense = self._repeat_kv_to_q_heads(v_dense)  # [T_kv, H]

            # reshape to BSH for FA
            T_kv = int(k_dense.shape[0])
            k_bsh = self._Reshape(k_dense, (1, T_kv, -1))
            v_bsh = self._Reshape(v_dense, (1, T_kv, -1))

            am, aql, akl = self._sanitize_fa_inputs(query, k_bsh, attn_mask, q_seq_lens, kv_end)
            self._dbg("Prefill-2+: FA(gathered); q=", aql, " kv_end=", akl,
                      " dtype=", query.dtype, " blk_sz=", block_size)

            _, _, _, ctx = self.flash_attention(query, k_bsh, v_bsh,
                                                None, None,
                                                None,   # padding_mask MUST be None
                                                am,     # attn_mask
                                                None,   # prefix
                                                aql, akl)
            return ctx

        # ---------------- DECODE / PA ----------------
        self._dbg("Decode/PA; dtype=", query.dtype)
        return self._run_paged(query, key_cache, value_cache, block_tables,
                               batch_valid_length, q_seq_lens, attn_mask)

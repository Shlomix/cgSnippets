# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Flash Attention Layer (BSH). Prefill-1: FA. Prefill-2+: FA w/ dense KV gather. Decode: PagedAttention."""
__all__ = ['FlashAttention']

import os
from mindspore import ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """
    BSH FlashAttention + PagedAttention bridge for parallel_core.Attention.

    Behavior:
      • First prefill (no prior ctx): FlashAttention (BSH).
      • Second+ prefill (has prior ctx): FlashAttention (BSH) over full prefix by gathering dense KV from paged cache.
        (PoC supports batch=1 for the gather; else we fallback to PA.)
      • Decode: PagedAttention.

    Notes & constraints:
      • Ascend FA requires padding_mask=None (we never pass it).
      • Q/K/V and masks must share float dtype (fp16 or bf16); we align dtypes per call.
      • Many kernels require sequence lengths multiple of 16; we check and fallback to PA unless forced.
      • Env toggles:
          MF_DEBUG_ATTENTION=1        -> prints at key steps (paths, lens, dtypes).
          MF_FORCE_FA_CHUNK=1         -> (default=1) attempt FA on 2nd+ prefill.
          MF_MANIFEST_FA_MISMATCH=1   -> force FA even if q_len!=kv_len or not x16 (to demonstrate failure).
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
        # Model params
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
        self._Tile = ops.Tile()
        self._Add = ops.Add()
        self._Mul = ops.Mul()
        self._Less = ops.Less()
        self._Mod = ops.Mod()
        self._ReduceSum = ops.ReduceSum(keep_dims=False)
        self._ReduceMax = ops.ReduceMax(keep_dims=False)
        self._Gather = ops.Gather()
        self._GatherD = ops.GatherD()
        self._FloorDiv = ops.FloorDiv()
        self._Minimum = ops.Minimum()

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

    # --------------------------- debug helper ---------------------------
    def _dbg(self, *args):
        if self.debug:
            self._Print(*args)

    # --------------------------- small helpers --------------------------
    def _all_x16(self, x_i32):
        rem = self._Mod(x_i32, ops.scalar_to_tensor(16, mstype.int32))
        s = self._ReduceSum(rem, 0)
        return s == ops.scalar_to_tensor(0, mstype.int32)

    def _repeat_kv_to_q_heads(self, dense_kv):
        """dense_kv: [T_kv, kv_heads*head_dim] -> [T_kv, head_num*head_dim] if kv_heads < head_num"""
        if self.kv_head_num is None or self.head_dim is None:
            return dense_kv
        if self.kv_head_num == self.head_num:
            return dense_kv
        rep = self.head_num // self.kv_head_num
        x = self._Reshape(dense_kv, (-1, self.kv_head_num, self.head_dim))
        x = self._Tile(x, (1, rep, 1))
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
                am = self._Tile(am, (B, 1, 1))         # [B,Sq,Sk]
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

    def _dense_kv_from_cache_b1(self, key_cache, value_cache, block_tables, kv_end_len_i32):
        """
        Gather dense KV from paged cache for B=1.
        key_cache shape: [num_blocks, block_size, kv_heads*head_dim]  or [num_blocks, block_size, kv_heads, head_dim]
        block_tables: [1, num_blocks] with physical block ids.
        kv_end_len_i32: scalar int32 (prefix length to include).
        Returns K_dense, V_dense as TH: [T_kv, H_kv] with dtype == cache dtype.
        """
        # infer shapes
        bt = block_tables[0]  # [num_blocks]
        num_blocks = int(bt.shape[0])
        shape = key_cache.shape
        if len(shape) == 4:
            nb, bs, kvh, hd = shape
            H_kv = kvh * hd
            kc_flat = self._Reshape(key_cache, (nb * bs, H_kv))
            vc_flat = self._Reshape(value_cache, (nb * bs, H_kv))
        elif len(shape) == 3:
            nb, bs, H_kv = shape
            kc_flat = self._Reshape(key_cache, (nb * bs, H_kv))
            vc_flat = self._Reshape(value_cache, (nb * bs, H_kv))
        else:
            self._dbg("Unexpected cache rank:", len(shape), "shape=", shape)
            # Best-effort reshape will trigger clear runtime error if wrong
            nb, bs = int(shape[0]), int(shape[1])
            H_kv = int(shape[-1])
            kc_flat = self._Reshape(key_cache, (nb * bs, H_kv))
            vc_flat = self._Reshape(value_cache, (nb * bs, H_kv))

        # how many blocks to cover kv_end_len
        kv_end = int(kv_end_len_i32.asnumpy().item())
        used_blocks = (kv_end + bs - 1) // bs
        used_blocks = min(used_blocks, num_blocks)

        # build token indices
        # block ids (physical)
        ids = bt[:used_blocks]  # [used_blocks], int32
        # offsets 0..bs-1
        offs = self._Range(ops.scalar_to_tensor(0, mstype.int32),
                           ops.scalar_to_tensor(bs, mstype.int32),
                           ops.scalar_to_tensor(1, mstype.int32))  # [bs]
        offs = self._ExpandDims(offs, 0)                            # [1,bs]
        # [used_blocks, bs] -> [used_blocks*bs]
        tok_idx = self._Reshape(self._Add(self._Mul(ids.view(-1, 1), ops.scalar_to_tensor(bs, mstype.int32)) + offs, 0),
                                (-1,))
        # last block may be partial
        tok_idx = tok_idx[:kv_end]                                  # [kv_end]

        k_dense = self._Gather(kc_flat, tok_idx, 0)                 # [kv_end, H_kv]
        v_dense = self._Gather(vc_flat, tok_idx, 0)
        return k_dense, v_dense, bs

    def _run_paged(self, query, key_cache, value_cache, block_tables, batch_valid_length, q_seq_lens, attn_mask):
        # Ensure PA mask dtype matches query/cache dtype
        want = (key_cache.dtype if key_cache is not None else query.dtype)
        if attn_mask is not None and attn_mask.dtype != want:
            attn_mask = self._Cast(attn_mask, want)
        if self.use_multi_latent_attention:
            return self.paged_attention(query, key_cache, key_cache,
                                        block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)
        return self.paged_attention(query, key_cache, value_cache,
                                    block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)

    # ------------------------------ forward ------------------------------
    def construct(self,
                  query,            # [B,Sq,H]
                  key,              # [B,Sq,H_kv] (current chunk kv)
                  value,            # [B,Sq,H_kv]
                  slot_mapping=None,
                  block_tables=None,
                  batch_valid_length=None,  # [B], context length BEFORE this chunk
                  context_lens_tensor=None,
                  q_seq_lens=None,          # [B] chunk sizes
                  actual_seq_qlen=None,     # [B] used by FA (first prefill)
                  actual_seq_kvlen=None,    # [B] used by FA (first prefill)
                  attn_mask=None,           # causal/prefix mask
                  padding_mask=None,        # MUST be None for FA on Ascend
                  prefix=None,
                  key_cache=None,
                  value_cache=None):
        """Forward process."""
        B, Sq, Hq = query.shape
        caches_exist = (key_cache is not None) and (value_cache is not None)
        target_dtype = key_cache.dtype if caches_exist else query.dtype

        # Align dtypes early
        if query.dtype != target_dtype:
            query = self._Cast(query, target_dtype)
        if key is not None and key.dtype != target_dtype:
            key = self._Cast(key, target_dtype)
        if value is not None and value.dtype != target_dtype:
            value = self._Cast(value, target_dtype)

        # Cache write for this step (needed by PA and by our gather later)
        if caches_exist and (slot_mapping is not None) and (not self.use_multi_latent_attention):
            if slot_mapping.dtype != mstype.int32:
                slot_mapping = self._Cast(slot_mapping, mstype.int32)
            # Reshape & write current chunk KV into cache
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # -------- PREFILL FLOW --------
        if self.is_prefill:
            padding_mask = None  # Ascend FA requirement

            # First prefill (no prior context): standard FA with current Q/K/V
            has_prior_ctx = (batch_valid_length is not None) and \
                            (int(self._ReduceMax(self._Cast(batch_valid_length, mstype.int32)).asnumpy()) > 0)
            if not has_prior_ctx:
                # use lengths if provided; else infer
                aql = actual_seq_qlen if actual_seq_qlen is not None else self._Cast(
                    ops.fill(mstype.int32, (B,), Sq), mstype.int32)
                akl = actual_seq_kvlen if actual_seq_kvlen is not None else self._Cast(
                    ops.fill(mstype.int32, (B,), Sq), mstype.int32)

                # x16 alignment check
                aligned = self._all_x16(self._Cast(aql, mstype.int32)) and self._all_x16(self._Cast(akl, mstype.int32))
                if not aligned and not self.manifest_mismatch:
                    self._dbg("Prefill-1: not x16 -> PA fallback; aql=", aql, " akl=", akl, " dtype=", query.dtype)
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

            # Second+ prefill (there is prior context)
            if not self.force_fa_chunk:
                self._dbg("Prefill-2+: forced off -> PA")
                return self._run_paged(query, key_cache, value_cache, block_tables,
                                       batch_valid_length, q_seq_lens, attn_mask)

            # PoC: support only B=1 for FA gather-from-cache
            if B != 1 and not self.manifest_mismatch:
                self._dbg("Prefill-2+: B!=1 -> PA fallback (PoC limitation)")
                return self._run_paged(query, key_cache, value_cache, block_tables,
                                       batch_valid_length, q_seq_lens, attn_mask)

            # Compute kv_end = prev_ctx + this_chunk
            if q_seq_lens is None:
                # infer from current tensor length
                q_seq_lens = ops.fill(mstype.int32, (B,), Sq)
            if q_seq_lens.dtype != mstype.int32:
                q_seq_lens = self._Cast(q_seq_lens, mstype.int32)
            if batch_valid_length.dtype != mstype.int32:
                batch_valid_length = self._Cast(batch_valid_length, mstype.int32)
            kv_end = self._Add(batch_valid_length, q_seq_lens)  # [B]

            # x16 alignment check (q and kv_end)
            aligned = self._all_x16(q_seq_lens) and self._all_x16(kv_end)
            if (not aligned) and (not self.manifest_mismatch):
                self._dbg("Prefill-2+: not x16 -> PA fallback; q=", q_seq_lens, " kv_end=", kv_end)
                return self._run_paged(query, key_cache, value_cache, block_tables,
                                       batch_valid_length, q_seq_lens, attn_mask)

            # Gather dense KV from cache for full prefix+chunk (B=1)
            k_dense, v_dense, block_size = self._dense_kv_from_cache_b1(
                key_cache, value_cache, self._Cast(block_tables, mstype.int32), kv_end[0]
            )  # TH [T_kv, H_kv]

            # Repeat KV heads to match query heads if needed
            k_dense = self._repeat_kv_to_q_heads(k_dense)  # TH [T_kv, H]
            v_dense = self._repeat_kv_to_q_heads(v_dense)  # TH [T_kv, H]

            # Reshape dense KV to BSH
            T_kv = int(k_dense.shape[0])
            k_bsh = self._Reshape(k_dense, (1, T_kv, -1))   # [1,Sk,H]
            v_bsh = self._Reshape(v_dense, (1, T_kv, -1))

            # Build FA lengths
            aql = q_seq_lens
            akl = kv_end

            am, aql, akl = self._sanitize_fa_inputs(query, k_bsh, attn_mask, aql, akl)
            self._dbg("Prefill-2+: FA(gathered); q=", aql, " kv_end=", akl,
                      " Hq=", Hq, " dtype=", query.dtype, " block_size=", block_size)

            # Call FA (BSH)
            _, _, _, ctx = self.flash_attention(query, k_bsh, v_bsh,
                                                None, None,
                                                None,   # padding_mask MUST be None
                                                am,     # attn_mask
                                                None,   # prefix
                                                aql, akl)
            return ctx

        # -------- DECODE / FALLBACK --------
        self._dbg("Decode/PA fallback; dtype=", query.dtype)
        return self._run_paged(query, key_cache, value_cache, block_tables,
                               batch_valid_length, q_seq_lens, attn_mask)

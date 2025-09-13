# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Flash Attention Layer with runtime x16 gating and safe fallback to PagedAttention."""
__all__ = ['FlashAttention']

import os
from mindspore import ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """
    first prefill:
        - if (S_q % 16 == 0 and S_kv % 16 == 0) → FlashAttention
        - else → PagedAttention (warm-up safe)
    2nd+ chunk prefill:
        - if MF_FORCE_FA_CHUNK=1 and x16-aligned (q_seq_lens & kv_end) → gather KV → FlashAttention
        - else → PagedAttention
    decode:
        - PagedAttention
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
                 input_layout="TH",
                 pa_kv_head_num=None,
                 pa_mla_v_dim=0):
        super().__init__()
        self.head_num = head_num
        self.hidden_size_per_attention_head = head_dim
        self.kv_head_num = kv_head_num
        self.sparse_mode = sparse_mode
        self.is_prefill = True
        self.input_layout = input_layout
        self.use_multi_latent_attention: bool = pa_mla_v_dim > 0

        # env toggles
        self.force_fa_chunk2p = os.getenv("MF_FORCE_FA_CHUNK", "1") == "1"
        self.debug            = os.getenv("MF_DEBUG_ATTENTION", "0") == "1"

        # primitives
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.input_layout,
            sparse_mode=self.sparse_mode,
        )
        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim
        )

        # util ops
        self._Print      = ops.Print()
        self._Range      = ops.Range()
        self._Reshape    = ops.Reshape()
        self._ExpandDims = ops.ExpandDims()
        self._Tile       = ops.Tile()
        self._Cast       = ops.Cast()
        self._ReduceMax  = ops.ReduceMax(keep_dims=False)
        self._ReduceAll  = ops.ReduceAll(keep_dims=False)
        self._Less       = ops.Less()
        self._Greater    = ops.Greater()
        self._Equal      = ops.Equal()
        self._Mul        = ops.Mul()
        self._Add        = ops.Add()
        self._FloorDiv   = ops.FloorDiv()
        self._Mod        = ops.Mod()
        self._Gather     = ops.Gather()
        self._GatherD    = ops.GatherD()
        self._Concat     = ops.Concat(axis=1)
        self._Shape      = ops.Shape()
        self._Fill       = ops.Fill()

    # ---------- small debug helper ----------
    def _dbg(self, *args):
        if self.debug:
            self._Print(*args)

    def _i32(self, x):
        return self._Cast(x, mstype.int32)

    # ---------- x16 checks ----------
    def _fa_ok_first_prefill(self, query, key):
        # use static integer shapes (no graph ops)
        sq = query.shape[1]
        sk = key.shape[1]
        ok = (sq % 16 == 0) and (sk % 16 == 0)
        self._dbg("FA.chk first_prefill sq/sk=", str(sq), "/", str(sk), " -> ", str(ok))
        return ok

    def _is_second_plus_chunk(self, q_seq_lens):
        if q_seq_lens is None:
            return ops.scalar_to_tensor(False, mstype.bool_)
        return self._Greater(self._ReduceMax(q_seq_lens),
                             ops.scalar_to_tensor(1, q_seq_lens.dtype))

    def _fa_ok_chunk2p(self, q_seq_lens, batch_valid_length):
        if (q_seq_lens is None) or (batch_valid_length is None):
            return ops.scalar_to_tensor(False, mstype.bool_)
        q = self._i32(q_seq_lens)
        b = self._i32(batch_valid_length)
        kv_end = self._Add(q, b)  # [B]
        q_ok  = self._ReduceAll(self._Equal(self._Mod(q,      ops.scalar_to_tensor(16, mstype.int32)),
                                            ops.scalar_to_tensor(0,  mstype.int32)))
        kv_ok = self._ReduceAll(self._Equal(self._Mod(kv_end, ops.scalar_to_tensor(16, mstype.int32)),
                                            ops.scalar_to_tensor(0,  mstype.int32)))
        ok = ops.LogicalAnd()(q_ok, kv_ok)
        self._dbg("FA.chk 2nd+ q=", q, " bvl=", b, " kv_end=", kv_end, " -> q_ok=", q_ok, " kv_ok=", kv_ok)
        return ok

    # ---------- gather contiguous KV from paged cache ----------
    def _gather_contiguous_kv(self, key_cache, value_cache, block_tables, batch_valid_length, q_seq_lens, bsz):
        block_tables       = self._i32(block_tables)
        batch_valid_length = self._i32(batch_valid_length)
        q_seq_lens         = self._i32(q_seq_lens)

        kv_end_len = self._Add(batch_valid_length, q_seq_lens)               # [B]
        max_kv_len = self._ReduceMax(kv_end_len)                             # scalar

        pos = self._Range(ops.scalar_to_tensor(0, mstype.int32),
                          max_kv_len,
                          ops.scalar_to_tensor(1, mstype.int32))            # [max_kv_len]
        pos_mat = self._Tile(self._ExpandDims(pos, 0), (bsz, 1))             # [B, max_kv_len]
        mask = self._Less(pos_mat, self._ExpandDims(kv_end_len, 1))          # [B, max_kv_len] bool

        BS = self._i32(ops.scalar_to_tensor(key_cache.shape[1], mstype.int32))
        blk_ids = self._FloorDiv(pos_mat, BS)
        offs    = self._Mod(pos_mat,      BS)
        phys_blk  = self._GatherD(block_tables, 1, blk_ids)                  # [B, max_kv_len]
        global_ix = self._Add(ops.Mul()(phys_blk, BS), offs)                 # [B, max_kv_len]
        flat_ix   = self._Reshape(global_ix, (-1,))                          # [B*max_kv_len]

        NB, BSv = key_cache.shape[0], key_cache.shape[1]
        if len(key_cache.shape) == 4:
            Hk, D = key_cache.shape[2], key_cache.shape[3]
            k_flat = self._Reshape(key_cache,   (NB * BSv, Hk * D))
            v_flat = self._Reshape(value_cache, (NB * BSv, Hk * D))
        else:
            HkD = key_cache.shape[2]
            k_flat = self._Reshape(key_cache,   (NB * BSv, HkD))
            v_flat = self._Reshape(value_cache, (NB * BSv, HkD))

        k_rows = self._Gather(k_flat, flat_ix, 0)                            # [B*max_kv_len, Hk*D]
        v_rows = self._Gather(v_flat, flat_ix, 0)
        k_full = self._Reshape(k_rows, (bsz, max_kv_len, -1))
        v_full = self._Reshape(v_rows, (bsz, max_kv_len, -1))

        # zero tail (safety)
        mask_f = self._Cast(self._ExpandDims(mask, 2), k_full.dtype)         # [B, max_kv_len, 1]
        k_contig = ops.Mul()(k_full, mask_f)
        v_contig = ops.Mul()(v_full, mask_f)

        self._dbg("FA.gather k/v:", str(k_contig.shape), "/", str(v_contig.shape),
                  " max_kv_len=", max_kv_len)
        return k_contig, v_contig

    # ---------- branch bodies (used only in non-prefill graph) ----------
    def _run_fa_chunk(self, query, k_contig, v_contig, padding_mask, attn_mask,
                      actual_seq_qlen, actual_seq_kvlen):
        self._dbg("PATH: FlashAttention (2nd+ chunk)")
        _, _, _, ctx = self.flash_attention(
            query, k_contig, v_contig,
            None, None, padding_mask, attn_mask,
            None, actual_seq_qlen, actual_seq_kvlen
        )
        return ctx

    def _run_paged(self, query, key_cache, value_cache, block_tables, batch_valid_length,
                   q_seq_lens, attn_mask):
        self._dbg("PATH: PagedAttention")
        if self.use_multi_latent_attention:
            return self.paged_attention(query, key_cache, key_cache,
                                        block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)
        return self.paged_attention(query, key_cache, value_cache,
                                    block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)

    # ---------- forward ----------
    def construct(self,
                  query,
                  key,
                  value,
                  slot_mapping=None,
                  block_tables=None,
                  batch_valid_length=None,
                  context_lens_tensor=None,
                  q_seq_lens=None,
                  actual_seq_qlen=None,
                  actual_seq_kvlen=None,
                  attn_mask=None,
                  padding_mask=None,
                  prefix=None,
                  key_cache=None,
                  value_cache=None):
        # Always write K/V into paged cache (unless MLA)
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        bsz = query.shape[0]
        self._dbg("ENTER FA: bsz=", str(bsz),
                  " q=", str(query.shape), " k=", str(key.shape), " v=", str(value.shape),
                  " is_prefill=", str(self.is_prefill),
                  " force_fa_chunk2p=", str(self.force_fa_chunk2p))

        # ---- FIRST PREFILL ----
        if self.is_prefill:
            # Warm-up safe: only use FA if lengths are x16; else Paged.
            if self._fa_ok_first_prefill(query, key):
                self._dbg("PATH: FlashAttention (first prefill)")
                _, _, _, ctx_prefill = self.flash_attention(
                    query, key, value,
                    None, None, padding_mask, attn_mask,
                    prefix, actual_seq_qlen, actual_seq_kvlen
                )
                return ctx_prefill
            else:
                self._dbg("PATH: PagedAttention (first prefill not x16)")
                return self._run_paged(query, key_cache, value_cache, block_tables,
                                       batch_valid_length, q_seq_lens, attn_mask)

        # ---- NON-PREFILL (decode or 2nd+ chunk prefill) ----
        is_chunk2p = self._is_second_plus_chunk(q_seq_lens)  # Tensor[bool]
        if self.force_fa_chunk2p and bool(is_chunk2p.asnumpy()):
            # Check x16 runtime gate for chunk2p
            if bool(self._fa_ok_chunk2p(q_seq_lens, batch_valid_length).asnumpy()):
                # Gather contiguous KV then run FA
                k_contig, v_contig = self._gather_contiguous_kv(
                    key_cache, value_cache, block_tables, batch_valid_length, q_seq_lens, bsz
                )
                return self._run_fa_chunk(query, k_contig, v_contig, padding_mask, attn_mask,
                                          actual_seq_qlen, actual_seq_kvlen)
            else:
                self._dbg("2nd+ not x16 → fallback to PagedAttention")

        # Decode or ineligible 2nd+ chunk → Paged
        return self._run_paged(query, key_cache, value_cache, block_tables,
                               batch_valid_length, q_seq_lens, attn_mask)

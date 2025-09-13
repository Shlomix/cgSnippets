# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Flash Attention Layer with runtime x16 gating and safe fallback to PagedAttention."""
__all__ = ['FlashAttention']

import os
import numpy as np

from mindspore import ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """
    Behavior summary
    ----------------
    first prefill:
        - if (S_q % 16 == 0 and S_kv % 16 == 0) → FlashAttention
        - else                                   → PagedAttention (warm-up safe)

    2nd+ chunk prefill:
        - if MF_FORCE_FA_CHUNK=1 and x16 aligned (q_seq_lens & (batch_valid_length + q_seq_lens))
            → gather contiguous KV (host-computed T_max as Python int) → FlashAttention
        - else
            → PagedAttention

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
        self._Equal      = ops.Equal()
        self._Mul        = ops.Mul()
        self._Add        = ops.Add()
        self._FloorDiv   = ops.FloorDiv()
        self._Mod        = ops.Mod()
        self._Gather     = ops.Gather()
        self._GatherD    = ops.GatherD()
        self._Concat     = ops.Concat(axis=1)

    # ---------- small debug helper ----------
    def _dbg(self, *args):
        if self.debug:
            self._Print(*args)

    def _i32(self, x):
        return self._Cast(x, mstype.int32)

    # ---------- x16 checks ----------
    def _fa_ok_first_prefill(self, query, key):
        # Use static integer shapes; these are Python ints in MindSpore
        sq = query.shape[1]
        sk = key.shape[1]
        ok = (sq % 16 == 0) and (sk % 16 == 0)
        self._dbg("FA.chk first_prefill sq/sk=", str(sq), "/", str(sk), " -> ", str(ok))
        return ok

    def _is_second_plus_chunk(self, q_seq_lens):
        if q_seq_lens is None:
            return False
        # Safe host decision (keeps warm-up graph clean)
        return int(np.max(q_seq_lens.asnumpy())) > 1

    def _fa_ok_chunk2p(self, q_seq_lens, batch_valid_length):
        if (q_seq_lens is None) or (batch_valid_length is None):
            return False
        q_np = q_seq_lens.asnumpy().astype(np.int64)
        b_np = batch_valid_length.asnumpy().astype(np.int64)
        kv_end_np = q_np + b_np
        q_ok  = np.all((q_np      % 16) == 0)
        kv_ok = np.all((kv_end_np % 16) == 0)
        self._dbg("FA.chk 2nd+ q=", str(q_np), " bvl=", str(b_np), " kv_end=", str(kv_end_np),
                  " -> q_ok=", str(q_ok), " kv_ok=", str(kv_ok))
        return bool(q_ok and kv_ok), int(np.max(kv_end_np))

    # ---------- gather contiguous KV from paged cache ----------
    # NOTE: max_kv_len_py and bsz_py are Python ints (NOT tensors), so Reshape receives ints
    def _gather_contiguous_kv(self, key_cache, value_cache,
                              block_tables, batch_valid_length, q_seq_lens,
                              bsz_py: int, max_kv_len_py: int):
        # Tensor forms (for index math & masking)
        block_tables_t       = self._i32(block_tables)
        batch_valid_length_t = self._i32(batch_valid_length)
        q_seq_lens_t         = self._i32(q_seq_lens)
        max_kv_len_t         = ops.scalar_to_tensor(max_kv_len_py, mstype.int32)

        kv_end_len = self._Add(batch_valid_length_t, q_seq_lens_t)           # [B]

        # positions [0..max_kv_len_py-1] as Tensor
        pos = self._Range(ops.scalar_to_tensor(0, mstype.int32),
                          max_kv_len_t,
                          ops.scalar_to_tensor(1, mstype.int32))             # [max_kv_len_py]
        pos_mat = self._Tile(self._ExpandDims(pos, 0), (bsz_py, 1))          # [B, max_kv_len_py]
        mask = self._Less(pos_mat, self._ExpandDims(kv_end_len, 1))          # [B, max_kv_len_py] bool

        # map logical positions → physical slots
        BS = int(key_cache.shape[1])                                        # block_size (Python int)
        blk_ids = self._FloorDiv(pos_mat, ops.scalar_to_tensor(BS, mstype.int32))
        offs    = self._Mod(pos_mat,      ops.scalar_to_tensor(BS, mstype.int32))
        phys_blk  = self._GatherD(block_tables_t, 1, blk_ids)                # [B, max_kv_len_py]
        global_ix = self._Add(self._Mul()(phys_blk, ops.scalar_to_tensor(BS, mstype.int32)), offs)
        flat_ix   = self._Reshape(global_ix, (-1,))                          # [B*max_kv_len_py]

        # flatten caches to [total_slots, Hk*D]
        NB, BSv = key_cache.shape[0], key_cache.shape[1]
        if len(key_cache.shape) == 4:
            Hk, D = key_cache.shape[2], key_cache.shape[3]
            k_flat = self._Reshape(key_cache,   (NB * BSv, Hk * D))
            v_flat = self._Reshape(value_cache, (NB * BSv, Hk * D))
        else:
            HkD = key_cache.shape[2]
            k_flat = self._Reshape(key_cache,   (NB * BSv, HkD))
            v_flat = self._Reshape(value_cache, (NB * BSv, HkD))

        # gather rows and reshape back with Python ints (no dynamic dims in tuple)
        k_rows = self._Gather(k_flat, flat_ix, 0)                            # [B*max_kv_len_py, Hk*D]
        v_rows = self._Gather(v_flat, flat_ix, 0)
        k_full = self._Reshape(k_rows, (bsz_py, max_kv_len_py, -1))
        v_full = self._Reshape(v_rows, (bsz_py, max_kv_len_py, -1))

        # zero tail (safety)
        mask_f = self._Cast(self._ExpandDims(mask, 2), k_full.dtype)         # [B, max_kv_len_py, 1]
        k_contig = self._Mul()(k_full, mask_f)
        v_contig = self._Mul()(v_full, mask_f)

        self._dbg("FA.gather k/v shapes:", str(k_contig.shape), "/", str(v_contig.shape),
                  " max_kv_len_py=", str(max_kv_len_py), " BS=", str(BS))
        return k_contig, v_contig

    # ---------- small wrappers ----------
    def _run_fa(self, query, key, value, padding_mask, attn_mask, actual_seq_qlen, actual_seq_kvlen, label):
        self._dbg("PATH:", label)
        _, _, _, ctx = self.flash_attention(
            query, key, value,
            None, None, padding_mask, attn_mask,
            None, actual_seq_qlen, actual_seq_kvlen
        )
        return ctx

    def _run_paged(self, query, key_cache, value_cache, block_tables, batch_valid_length, q_seq_lens, attn_mask):
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

        bsz_py = int(query.shape[0])  # Python int for reshape
        self._dbg("ENTER FA: bsz=", str(bsz_py),
                  " q=", str(query.shape), " k=", str(key.shape), " v=", str(value.shape),
                  " is_prefill=", str(self.is_prefill),
                  " force_fa_chunk2p=", str(self.force_fa_chunk2p))

        # ---- FIRST PREFILL ----
        if self.is_prefill:
            # Warm-up safe: only use FA if lengths are x16; else Paged.
            if self._fa_ok_first_prefill(query, key):
                return self._run_fa(query, key, value, padding_mask, attn_mask,
                                    actual_seq_qlen, actual_seq_kvlen,
                                    "FlashAttention (first prefill)")
            else:
                return self._run_paged(query, key_cache, value_cache, block_tables,
                                       batch_valid_length, q_seq_lens, attn_mask)

        # ---- NON-PREFILL (decode or 2nd+ chunk prefill) ----
        if self.force_fa_chunk2p and self._is_second_plus_chunk(q_seq_lens):
            ok, max_kv_len_py = self._fa_ok_chunk2p(q_seq_lens, batch_valid_length)
            if ok:
                # We compute Python ints here (host side) and pass them to the gather helper,
                # so that all Reshape target shapes are tuples of Python ints (no tensors).
                k_contig, v_contig = self._gather_contiguous_kv(
                    key_cache, value_cache,
                    block_tables, batch_valid_length, q_seq_lens,
                    bsz_py=bsz_py, max_kv_len_py=max_kv_len_py
                )
                return self._run_fa(query, k_contig, v_contig, padding_mask, attn_mask,
                                    actual_seq_qlen, actual_seq_kvlen,
                                    "FlashAttention (2nd+ chunk)")
            else:
                self._dbg("2nd+ not x16 → fallback to PagedAttention")

        # Decode or ineligible 2nd+ chunk → Paged
        return self._run_paged(query, key_cache, value_cache, block_tables,
                               batch_valid_length, q_seq_lens, attn_mask)

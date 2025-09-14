# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Flash Attention Layer (BSH-only, minimal, fp16 cache contract)."""
__all__ = ['FlashAttention']

import os
import numpy as np

from mindspore import ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    def __init__(self,
                 head_num,
                 head_dim=None,
                 kv_head_num=None,
                 keep_prob=1.0,
                 scale_value=1.0,
                 pre_tokens=2147483647,
                 next_tokens=2147483647,
                 sparse_mode=0,
                 input_layout="BSH",   # BSH only
                 pa_kv_head_num=None,
                 pa_mla_v_dim=0):
        super().__init__()
        self.head_num = head_num
        self.hidden_size_per_attention_head = head_dim
        self.kv_head_num = kv_head_num
        self.sparse_mode = sparse_mode
        self.is_prefill = True
        self.use_multi_latent_attention: bool = pa_mla_v_dim > 0

        # env toggles
        self.force_fa_chunk2p = os.getenv("MF_FORCE_FA_CHUNK", "1") == "1"
        self.debug            = os.getenv("MF_DEBUG_ATTENTION", "0") == "1"

        # ops
        self._Print      = ops.Print()
        self._Cast       = ops.Cast()
        self._Reshape    = ops.Reshape()
        self._ExpandDims = ops.ExpandDims()
        self._Tile       = ops.Tile()
        self._Range      = ops.Range()
        self._Add        = ops.Add()
        self._Mul        = ops.Mul()
        self._Less       = ops.Less()
        self._FloorDiv   = ops.FloorDiv()
        self._Mod        = ops.Mod()
        self._Gather     = ops.Gather()
        self._GatherD    = ops.GatherD()

        # kernels
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout="BSH",        # <- BSH layout
            sparse_mode=self.sparse_mode,
        )
        self.paged_attention = ops.auto_generate.PagedAttention(
            head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim
        )
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

    # ------------------- helpers -------------------
    def _dbg(self, *args):
        if self.debug:
            self._Print(*args)

    def _i32(self, x):
        return self._Cast(x, mstype.int32)

    def _fa_ok_first_prefill(self, query, key):
        # query/key are [B, S, H] here
        sq = int(query.shape[1])
        sk = int(key.shape[1])
        ok = (sq % 16 == 0) and (sk % 16 == 0)
        self._dbg("chk first_prefill: S_q=", str(sq), " S_kv=", str(sk), " x16=", str(ok))
        return ok

    def _is_second_plus_chunk(self, q_seq_lens):
        if q_seq_lens is None:
            return False
        # “second+” if any request contributes more than 1 token this step
        return int(np.max(q_seq_lens.asnumpy())) > 1

    def _fa_ok_chunk2p(self, q_seq_lens, batch_valid_length):
        if (q_seq_lens is None) or (batch_valid_length is None):
            return False, 0
        q_np = q_seq_lens.asnumpy().astype(np.int64)      # per-request new tokens
        b_np = batch_valid_length.asnumpy().astype(np.int64)  # context (cached) per request
        kv_end = q_np + b_np
        q_ok  = np.all((q_np   % 16) == 0)
        kv_ok = np.all((kv_end % 16) == 0)
        self._dbg("chk 2nd+: q=", str(q_np), " bvl=", str(b_np), " kv_end=", str(kv_end),
                  " x16_q=", str(q_ok), " x16_kv=", str(kv_ok))
        return bool(q_ok and kv_ok), int(np.max(kv_end))

    def _gather_contiguous_kv(self, key_cache, value_cache,
                              block_tables, batch_valid_length, q_seq_lens,
                              bsz_py: int, max_kv_len_py: int):
        """
        Build contiguous KV windows [0..kv_end) per request out of paged cache.

        Inputs:
          key_cache/value_cache: [num_blocks, block_size, H] or [num_blocks, block_size, n_kv_head, head_dim]
          block_tables: [B, num_blocks_per_req] int32 indices into key_cache rows
          batch_valid_length: [B] int32
          q_seq_lens: [B] int32
          max_kv_len_py: python int, max kv_end among batch

        Output:
          k_contig / v_contig: [B, max_kv_len_py, H] (zero-padded on the right)
        """
        block_tables = self._i32(block_tables)
        bvl = self._i32(batch_valid_length)
        qsl = self._i32(q_seq_lens)
        tmax = ops.scalar_to_tensor(max_kv_len_py, mstype.int32)

        pos = self._Range(ops.scalar_to_tensor(0, mstype.int32), tmax, ops.scalar_to_tensor(1, mstype.int32))
        pos_mat = self._Tile(self._ExpandDims(pos, 0), (bsz_py, 1))              # [B, T_max]
        kv_end = self._Add(bvl, qsl)                                             # [B]
        mask = self._Less(pos_mat, self._ExpandDims(kv_end, 1))                  # [B, T_max] bool

        BS = int(key_cache.shape[1])                                             # block_size
        BS_t = ops.scalar_to_tensor(BS, mstype.int32)
        blk_ids = self._FloorDiv(pos_mat, BS_t)                                  # [B, T_max]
        offs    = self._Mod(pos_mat,      BS_t)                                  # [B, T_max]
        phys_blk  = self._GatherD(block_tables, 1, blk_ids)                      # [B, T_max]
        global_ix = self._Add(self._Mul(phys_blk, BS_t), offs)                   # [B, T_max]
        flat_ix   = self._Reshape(global_ix, (-1,))                              # [B*T_max]

        NB, BSv = key_cache.shape[0], key_cache.shape[1]
        if len(key_cache.shape) == 4:
            Hk, D = key_cache.shape[2], key_cache.shape[3]
            k_flat = self._Reshape(key_cache,   (NB * BSv, Hk * D))
            v_flat = self._Reshape(value_cache, (NB * BSv, Hk * D))
        else:
            HkD = int(key_cache.shape[2])
            k_flat = self._Reshape(key_cache,   (NB * BSv, HkD))
            v_flat = self._Reshape(value_cache, (NB * BSv, HkD))

        k_rows = self._Gather(k_flat, flat_ix, 0)                                 # [B*T_max, H]
        v_rows = self._Gather(v_flat, flat_ix, 0)
        k_full = self._Reshape(k_rows, (bsz_py, max_kv_len_py, -1))               # [B, T_max, H]
        v_full = self._Reshape(v_rows, (bsz_py, max_kv_len_py, -1))

        mask_f = self._Cast(self._ExpandDims(mask, 2), k_full.dtype)              # [B, T_max, 1]
        k_contig = self._Mul(k_full, mask_f)
        v_contig = self._Mul(v_full, mask_f)
        return k_contig, v_contig

    def _run_fa_bsh(self, q_bsh, k_bsh, v_bsh,
                    padding_mask, attn_mask, actual_seq_qlen, actual_seq_kvlen):
        # BSH expects 3-D [B, S, H]
        _, _, _, ctx = self.flash_attention(
            q_bsh, k_bsh, v_bsh,
            None, None, padding_mask, attn_mask,
            None, actual_seq_qlen, actual_seq_kvlen
        )
        return ctx

    def _run_paged(self, query, key_cache, value_cache, block_tables, batch_valid_length, q_seq_lens, attn_mask):
        if self.use_multi_latent_attention:
            return self.paged_attention(query, key_cache, key_cache,
                                        block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)
        return self.paged_attention(query, key_cache, value_cache,
                                    block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)

    # ------------------- forward -------------------
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

        # Contract: caches (if exist) are float16.
        # If caches are present → cast K/V to fp16 and write cache.
        # If caches are None → cast Q/K/V to fp16 and skip cache write.
        caches_exist = (key_cache is not None) and (value_cache is not None)

        if not self.use_multi_latent_attention and not caches_exist:
            # cache is None → cast q/k/v to fp16 and skip cache write
            if query is not None and query.dtype != mstype.float16:
                query = self._Cast(query, mstype.float16)
            if key is not None and key.dtype != mstype.float16:
                key = self._Cast(key, mstype.float16)
            if value is not None and value.dtype != mstype.float16:
                value = self._Cast(value, mstype.float16)
        else:
            # slot_mapping must be int32; in-place cache write expects src dtype == cache dtype (fp16)
            if slot_mapping is not None and slot_mapping.dtype != mstype.int32:
                slot_mapping = self._Cast(slot_mapping, mstype.int32)

            if key is not None and key.dtype != mstype.float16:
                key = self._Cast(key, mstype.float16)
            if value is not None and value.dtype != mstype.float16:
                value = self._Cast(value, mstype.float16)

            # write into paged cache
            if not self.use_multi_latent_attention:
                self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # keep Q/K/V consistent for FA path (prefer fp16)
        if key is not None and query is not None and query.dtype != key.dtype:
            query = self._Cast(query, key.dtype)
        if value is not None and key is not None and value.dtype != key.dtype:
            value = self._Cast(value, key.dtype)

        # -------- FIRST PREFILL --------
        if self.is_prefill:
            if self._fa_ok_first_prefill(query, key):
                # BSH FlashAttention
                return self._run_fa_bsh(query, key, value, padding_mask, attn_mask,
                                        actual_seq_qlen, actual_seq_kvlen)
            else:
                # warm-up safe fallback
                return self._run_paged(query, key_cache, value_cache, block_tables,
                                       batch_valid_length, q_seq_lens, attn_mask)

        # -------- 2ND+ CHUNK PREFILL or DECODE --------
        if self.force_fa_chunk2p and self._is_second_plus_chunk(q_seq_lens):
            ok, max_kv_len_py = self._fa_ok_chunk2p(q_seq_lens, batch_valid_length)
            if ok and caches_exist:
                bsz_py = int(query.shape[0])
                k_contig, v_contig = self._gather_contiguous_kv(
                    key_cache, value_cache, block_tables, batch_valid_length, q_seq_lens,
                    bsz_py=bsz_py, max_kv_len_py=max_kv_len_py
                )
                # Ensure gathered KV matches Q dtype (fp16)
                if k_contig.dtype != query.dtype:
                    k_contig = self._Cast(k_contig, query.dtype)
                    v_contig = self._Cast(v_contig, query.dtype)
                return self._run_fa_bsh(query, k_contig, v_contig, padding_mask, attn_mask,
                                        actual_seq_qlen, actual_seq_kvlen)

        # decode or fallback
        return self._run_paged(query, key_cache, value_cache, block_tables,
                               batch_valid_length, q_seq_lens, attn_mask)

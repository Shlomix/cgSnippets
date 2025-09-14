# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Flash Attention Layer (BSH-only, simple 2nd+ chunk FA PoC)."""
__all__ = ['FlashAttention']

import os
from mindspore import ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """Minimal BSH-only FlashAttention + PagedAttention wrapper.

    Behavior:
      • First prefill: try FlashAttention if (actual_seq_qlen % 16 == 0) & (actual_seq_kvlen % 16 == 0)
        else fallback to PagedAttention.
      • 2nd+ chunk prefill: if MF_FORCE_FA_CHUNK=1 and per-request
        (q_seq_lens % 16 == 0) & ((batch_valid_length + q_seq_lens) % 16 == 0),
        gather a contiguous KV window from cache and run FlashAttention;
        else PagedAttention.
      • Decode: PagedAttention.

    Assumptions:
      • KV cache (if present) is fp16. We cast K/V to fp16 before cache write.
      • For FA calls we pass BSH tensors and sanitize masks to the kernel’s
        expected 7-input signature: (q, k, v, padding_mask, attn_mask,
        actual_seq_qlen, actual_seq_kvlen). No None inputs.
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

        # simple toggles
        self.force_fa_chunk2p = os.getenv("MF_FORCE_FA_CHUNK", "1") == "1"
        self.debug             = os.getenv("MF_DEBUG_ATTENTION", "0") == "1"

        # light wrappers
        self._Print      = ops.Print()
        self._Cast       = ops.Cast()
        self._Reshape    = ops.Reshape()
        self._ExpandDims = ops.ExpandDims()
        self._Tile       = ops.Tile()
        self._Range      = ops.Range()
        self._Add        = ops.Add()
        self._Mul        = ops.Mul()
        self._Less       = ops.Less()
        self._Mod        = ops.Mod()
        self._ReduceSum  = ops.ReduceSum(keep_dims=False)
        self._Greater    = ops.Greater()
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
            input_layout="BSH",        # BSH layout only
            sparse_mode=self.sparse_mode,
        )
        self.paged_attention = ops.auto_generate.PagedAttention(
            head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim
        )
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

    # ---------------- debug ----------------
    def _dbg(self, *args):
        if self.debug:
            self._Print(*args)

    # --------------- small helpers ---------------
    def _all_x16(self, x_i32):
        """Return scalar bool(tensor) true if all elements %16==0."""
        # sum(mod(x,16)) == 0  -> all divisible by 16
        rem = self._Mod(x_i32, ops.scalar_to_tensor(16, mstype.int32))
        s = self._ReduceSum(rem, 0)
        return s == ops.scalar_to_tensor(0, mstype.int32)

    def _is_second_plus_chunk(self, q_seq_lens):
        """Any request adds more than one token in this step?"""
        # max(q_seq_lens) > 1  -> some request contributes more than one token
        return self._Greater(ops.ReduceMax(keep_dims=False)(q_seq_lens),
                             ops.scalar_to_tensor(1, mstype.int32))

    def _sanitize_masks_and_lens(self, q, k, padding_mask, attn_mask,
                                 actual_seq_qlen, actual_seq_kvlen):
        """Make non-None masks/lens for FA call; cast dtypes/shapes as needed."""
        B = int(q.shape[0])
        S_q = int(q.shape[1])
        S_k = int(k.shape[1])

        # padding_mask -> uint8, shape [B, S_q] (zeros if absent)
        if padding_mask is None:
            padding_mask = ops.zeros((B, S_q), mstype.uint8)
        elif padding_mask.dtype != mstype.uint8:
            padding_mask = self._Cast(padding_mask, mstype.uint8)

        # attn_mask -> uint8, shape [B, S_q, S_k]
        # If it comes as float with big negatives for masked, convert: (attn < 0) -> 1 else 0
        if attn_mask is None:
            attn_mask = ops.zeros((B, S_q, S_k), mstype.uint8)
        else:
            if attn_mask.dtype != mstype.uint8:
                zeros_like = ops.zeros_like(attn_mask)
                attn_mask = self._Cast(self._Less(attn_mask, zeros_like), mstype.uint8)

            # If it was 2D [S_q, S_k] or [1, S_q, S_k], expand to [B, ...]
            if len(attn_mask.shape) == 2:
                attn_mask = self._ExpandDims(attn_mask, 0)            # [1, S_q, S_k]
                attn_mask = self._Tile(attn_mask, (B, 1, 1))
            elif len(attn_mask.shape) == 3 and attn_mask.shape[0] == 1 and B > 1:
                attn_mask = self._Tile(attn_mask, (B, 1, 1))

        # actual_seq_{q,kv}len -> int32 shape [B] (defaults to [S_q] / [S_k] if None)
        if actual_seq_qlen is None:
            actual_seq_qlen = ops.fill(mstype.int32, (B,), S_q)
        elif actual_seq_qlen.dtype != mstype.int32:
            actual_seq_qlen = self._Cast(actual_seq_qlen, mstype.int32)

        if actual_seq_kvlen is None:
            actual_seq_kvlen = ops.fill(mstype.int32, (B,), S_k)
        elif actual_seq_kvlen.dtype != mstype.int32:
            actual_seq_kvlen = self._Cast(actual_seq_kvlen, mstype.int32)

        return padding_mask, attn_mask, actual_seq_qlen, actual_seq_kvlen

    def _gather_contiguous_kv(self, key_cache, value_cache,
                              block_tables, batch_valid_length, q_seq_lens,
                              max_kv_len_py: int):
        """
        Build contiguous KV windows [0..kv_end) per request out of paged cache.

        Inputs:
          key_cache/value_cache: [num_blocks, block_size, H] or [num_blocks, block_size, n_kv_head, head_dim]
          block_tables: [B, num_blocks_per_req] int32 indices into key_cache rows
          batch_valid_length: [B] int32
          q_seq_lens: [B] int32
          max_kv_len_py: python int, max kv_end among batch

        Output:
          k_contig / v_contig: [B, max_kv_len_py, H] (zero-padded)
        """
        B = int(block_tables.shape[0])
        BS = int(key_cache.shape[1])
        BS_t = ops.scalar_to_tensor(BS, mstype.int32)

        # Positions we want to gather per request.
        tmax = ops.scalar_to_tensor(max_kv_len_py, mstype.int32)
        pos = self._Range(ops.scalar_to_tensor(0, mstype.int32), tmax, ops.scalar_to_tensor(1, mstype.int32))
        pos_mat = self._Tile(self._ExpandDims(pos, 0), (B, 1))                    # [B, T_max]

        kv_end = self._Add(batch_valid_length, q_seq_lens)                         # [B]
        mask = self._Less(pos_mat, self._ExpandDims(kv_end, 1))                   # [B, T_max]

        blk_ids = ops.FloorDiv()(pos_mat, BS_t)                                   # [B, T_max]
        offs    = self._Mod(pos_mat,      BS_t)                                   # [B, T_max]
        phys_blk  = self._GatherD(block_tables, 1, blk_ids)                       # [B, T_max]
        global_ix = self._Add(ops.Mul()(phys_blk, BS_t), offs)                    # [B, T_max]
        flat_ix   = self._Reshape(global_ix, (-1,))                               # [B*T_max]

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
        k_full = self._Reshape(k_rows, (B, max_kv_len_py, -1))                    # [B, T_max, H]
        v_full = self._Reshape(v_rows, (B, max_kv_len_py, -1))

        mask_f = self._Cast(self._ExpandDims(mask, 2), k_full.dtype)              # [B, T_max, 1]
        k_contig = self._Mul(k_full, mask_f)
        v_contig = self._Mul(v_full, mask_f)
        return k_contig, v_contig

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
        """Forward process of the FlashAttention."""

        # ---- Cache / dtype setup ----
        caches_exist = (key_cache is not None) and (value_cache is not None)

        # Contract: caches (if exist) are fp16; write in fp16; slot_mapping int32.
        if caches_exist:
            if slot_mapping is not None and slot_mapping.dtype != mstype.int32:
                slot_mapping = self._Cast(slot_mapping, mstype.int32)
            if key is not None and key.dtype != mstype.float16:
                key = self._Cast(key, mstype.float16)
            if value is not None and value.dtype != mstype.float16:
                value = self._Cast(value, mstype.float16)
            if not self.use_multi_latent_attention:
                self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
        else:
            # no cache: make sure q/k/v are fp16 for FA path
            if query is not None and query.dtype != mstype.float16:
                query = self._Cast(query, mstype.float16)
            if key is not None and key.dtype != mstype.float16:
                key = self._Cast(key, mstype.float16)
            if value is not None and value.dtype != mstype.float16:
                value = self._Cast(value, mstype.float16)

        # keep q/k/v dtypes aligned
        if key is not None and query is not None and query.dtype != key.dtype:
            query = self._Cast(query, key.dtype)
        if value is not None and key is not None and value.dtype != key.dtype:
            value = self._Cast(value, key.dtype)

        # ---------------- FIRST PREFILL ----------------
        if self.is_prefill:
            # We rely on provided actual_seq_* to test x16 (robust in graph mode).
            if (actual_seq_qlen is not None) and (actual_seq_kvlen is not None) \
               and self._all_x16(self._Cast(actual_seq_qlen, mstype.int32)) \
               and self._all_x16(self._Cast(actual_seq_kvlen, mstype.int32)):
                pm, am, aql, akl = self._sanitize_masks_and_lens(
                    query, key, padding_mask, attn_mask, actual_seq_qlen, actual_seq_kvlen
                )
                self._dbg("FA first prefill: aql=", aql, " akl=", akl)
                # 7-input BSH call
                _, _, _, ctx = self.flash_attention(query, key, value, pm, am, aql, akl)
                return ctx
            # fallback (warm-up etc.)
            self._dbg("PA first prefill fallback")
            return self._run_paged(query, key_cache, value_cache, block_tables,
                                   batch_valid_length, q_seq_lens, attn_mask)

        # --------------- 2ND+ CHUNK or DECODE ---------------
        # Try 2nd+ chunk FA (contiguous gather) when enabled & aligned.
        if self.force_fa_chunk2p and (q_seq_lens is not None) and (batch_valid_length is not None):
            is_second_plus = self._is_second_plus_chunk(q_seq_lens)
            kv_end = self._Add(self._Cast(batch_valid_length, mstype.int32),
                               self._Cast(q_seq_lens, mstype.int32))
            aligned_q  = self._all_x16(self._Cast(q_seq_lens, mstype.int32))
            aligned_kv = self._all_x16(kv_end)
            if is_second_plus and aligned_q and aligned_kv and caches_exist:
                max_kv_len_py = int(kv_end.max().asnumpy())  # shape scalar OK to read
                k_contig, v_contig = self._gather_contiguous_kv(
                    key_cache, value_cache, self._Cast(block_tables, mstype.int32),
                    self._Cast(batch_valid_length, mstype.int32),
                    self._Cast(q_seq_lens, mstype.int32),
                    max_kv_len_py=max_kv_len_py
                )
                # Make masks/lens for FA
                pm, am, aql, akl = self._sanitize_masks_and_lens(
                    query, k_contig, padding_mask, attn_mask, actual_seq_qlen, kv_end
                )
                self._dbg("FA 2nd+ chunk: q_len=", aql, " kv_len=", akl)
                _, _, _, ctx = self.flash_attention(query, k_contig, v_contig, pm, am, aql, akl)
                return ctx

        # decode or fallback path
        self._dbg("PA decode/fallback")
        return self._run_paged(query, key_cache, value_cache, block_tables,
                               batch_valid_length, q_seq_lens, attn_mask)

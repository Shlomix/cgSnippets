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
"""Flash Attention Layer (BSH-only, with strict dtype alignment for FA/PA)."""
__all__ = ['FlashAttention']

import os
from mindspore import ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """
    Minimal BSH FlashAttention + PagedAttention wrapper with robust dtype handling.

    Key rules:
      • Determine a single target dtype per call:
          - If caches exist -> cache dtype (key_cache.dtype).
          - Else -> query.dtype.
      • Cast q/k/v to target dtype.
      • For cache write: cast key/value to cache dtype before reshape_and_cache.
      • Cast masks (padding_mask, attn_mask) to the same float dtype as query for FA,
        and to target dtype for PA.
      • First prefill: try FA if lengths are 16-aligned, else PA fallback.
      • Second+ chunk prefill (if MF_FORCE_FA_CHUNK=1): gather contiguous KV and try FA
        if aligned; else PA.
      • Decode: PA.

    Environment toggles:
      MF_FORCE_FA_CHUNK=1  -> try FA on 2nd+ chunk prefill (default=1)
      MF_DEBUG_ATTENTION=1 -> Print debug tensors
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
            raise ValueError("This branch expects input_layout='BSH'.")
        self.head_num = head_num
        self.hidden_size_per_attention_head = head_dim
        self.kv_head_num = kv_head_num
        self.sparse_mode = sparse_mode
        self.is_prefill = True
        self.use_multi_latent_attention: bool = pa_mla_v_dim > 0

        # env toggles
        self.force_fa_chunk2p = os.getenv("MF_FORCE_FA_CHUNK", "1") == "1"
        self.debug             = os.getenv("MF_DEBUG_ATTENTION", "0") == "1"

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
        self._Mod        = ops.Mod()
        self._ReduceSum  = ops.ReduceSum(keep_dims=False)
        self._Greater    = ops.Greater()
        self._ReduceMax  = ops.ReduceMax(keep_dims=False)
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
            input_layout="BSH",
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

    # ---------------- helpers ----------------
    def _all_x16(self, x_i32):
        rem = self._Mod(x_i32, ops.scalar_to_tensor(16, mstype.int32))
        s = self._ReduceSum(rem, 0)
        return s == ops.scalar_to_tensor(0, mstype.int32)

    def _is_second_plus_chunk(self, q_seq_lens):
        return self._Greater(self._ReduceMax(q_seq_lens), ops.scalar_to_tensor(1, mstype.int32))

    def _sanitize_fa_masks(self, q, k, padding_mask, attn_mask, aql, akl):
        """Masks for FA: cast to q.dtype, fix shapes to [B,S] and [B,Sq,Sk]."""
        B = int(q.shape[0]); Sq = int(q.shape[1]); Sk = int(k.shape[1])
        qdtype = q.dtype

        if padding_mask is None:
            padding_mask = ops.zeros((B, Sq), qdtype)
        else:
            if len(padding_mask.shape) == 1 and padding_mask.shape[0] == B:
                padding_mask = self._Tile(self._ExpandDims(padding_mask, 1), (1, Sq))
            if padding_mask.dtype != qdtype:
                padding_mask = self._Cast(padding_mask, qdtype)

        if attn_mask is None:
            attn_mask = ops.zeros((B, Sq, Sk), qdtype)
        else:
            if attn_mask.dtype != qdtype:
                attn_mask = self._Cast(attn_mask, qdtype)
            if len(attn_mask.shape) == 2:
                attn_mask = self._ExpandDims(attn_mask, 0)
            if attn_mask.shape[0] == 1 and B > 1:
                attn_mask = self._Tile(attn_mask, (B, 1, 1))

        if aql is None:
            aql = ops.fill(mstype.int32, (B,), Sq)
        elif aql.dtype != mstype.int32:
            aql = self._Cast(aql, mstype.int32)

        if akl is None:
            akl = ops.fill(mstype.int32, (B,), Sk)
        elif akl.dtype != mstype.int32:
            akl = self._Cast(akl, mstype.int32)

        return padding_mask, attn_mask, aql, akl

    def _gather_contiguous_kv(self, key_cache, value_cache, block_tables,
                              batch_valid_length, q_seq_lens, max_kv_len_py: int):
        """Build contiguous KV [B, T_max, H] from paged cache, zero-pad beyond kv_end."""
        B = int(block_tables.shape[0])
        BS = int(key_cache.shape[1])
        BS_t = ops.scalar_to_tensor(BS, mstype.int32)

        tmax = ops.scalar_to_tensor(max_kv_len_py, mstype.int32)
        pos = self._Range(ops.scalar_to_tensor(0, mstype.int32), tmax, ops.scalar_to_tensor(1, mstype.int32))
        pos_mat = self._Tile(self._ExpandDims(pos, 0), (B, 1))  # [B, T_max]

        kv_end = self._Add(batch_valid_length, q_seq_lens)       # [B]
        mask = self._Less(pos_mat, self._ExpandDims(kv_end, 1))  # [B, T_max]

        blk_ids = ops.FloorDiv()(pos_mat, BS_t)
        offs    = self._Mod(pos_mat,      BS_t)
        phys_blk  = self._GatherD(block_tables, 1, blk_ids)
        global_ix = self._Add(ops.Mul()(phys_blk, BS_t), offs)
        flat_ix   = self._Reshape(global_ix, (-1,))

        NB, BSv = key_cache.shape[0], key_cache.shape[1]
        if len(key_cache.shape) == 4:
            Hk, D = key_cache.shape[2], key_cache.shape[3]
            k_flat = self._Reshape(key_cache,   (NB * BSv, Hk * D))
            v_flat = self._Reshape(value_cache, (NB * BSv, Hk * D))
        else:
            HkD = int(key_cache.shape[2])
            k_flat = self._Reshape(key_cache,   (NB * BSv, HkD))
            v_flat = self._Reshape(value_cache, (NB * BSv, HkD))

        k_rows = self._Gather(k_flat, flat_ix, 0)
        v_rows = self._Gather(v_flat, flat_ix, 0)
        k_full = self._Reshape(k_rows, (B, max_kv_len_py, -1))
        v_full = self._Reshape(v_rows, (B, max_kv_len_py, -1))

        mask_f = self._Cast(self._ExpandDims(mask, 2), k_full.dtype)
        k_contig = self._Mul(k_full, mask_f)
        v_contig = self._Mul(v_full, mask_f)
        return k_contig, v_contig

    def _run_paged(self, query, key_cache, value_cache, block_tables,
                   batch_valid_length, q_seq_lens, attn_mask):
        if self.use_multi_latent_attention:
            return self.paged_attention(query, key_cache, key_cache,
                                        block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)
        return self.paged_attention(query, key_cache, value_cache,
                                    block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)

    # ---------------- forward ----------------
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
        """Forward process."""
        caches_exist = (key_cache is not None) and (value_cache is not None)
        # Decide one target dtype for this call.
        target_dtype = key_cache.dtype if caches_exist else query.dtype

        # Align q/k/v to target dtype upfront.
        if query is not None and query.dtype != target_dtype:
            query = self._Cast(query, target_dtype)
        if key is not None and key.dtype != target_dtype:
            key = self._Cast(key, target_dtype)
        if value is not None and value.dtype != target_dtype:
            value = self._Cast(value, target_dtype)

        # Cache write path (must match cache dtype).
        if caches_exist:
            if slot_mapping is not None and slot_mapping.dtype != mstype.int32:
                slot_mapping = self._Cast(slot_mapping, mstype.int32)
            if key is not None and key.dtype != key_cache.dtype:
                key = self._Cast(key, key_cache.dtype)
            if value is not None and value.dtype != value_cache.dtype:
                value = self._Cast(value, value_cache.dtype)
            if not self.use_multi_latent_attention:
                self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # ---------- FIRST PREFILL ----------
        if self.is_prefill:
            if (actual_seq_qlen is not None) and (actual_seq_kvlen is not None) \
               and self._all_x16(self._Cast(actual_seq_qlen, mstype.int32)) \
               and self._all_x16(self._Cast(actual_seq_kvlen, mstype.int32)):
                # FA first prefill
                pm, am, aql, akl = self._sanitize_fa_masks(query, key, padding_mask, attn_mask,
                                                           actual_seq_qlen, actual_seq_kvlen)
                self._dbg("FA first prefill; dtype=", query.dtype, " aql=", aql, " akl=", akl)
                _, _, _, ctx = self.flash_attention(query, key, value, pm, am, aql, akl)
                return ctx
            # fallback to PA
            self._dbg("PA first prefill fallback; dtype=", target_dtype)
            # Ensure mask matches target dtype for PA
            if attn_mask is not None and attn_mask.dtype != target_dtype:
                attn_mask = self._Cast(attn_mask, target_dtype)
            return self._run_paged(query, key_cache, value_cache, block_tables,
                                   batch_valid_length, q_seq_lens, attn_mask)

        # ---------- 2ND+ CHUNK or DECODE ----------
        if self.force_fa_chunk2p and (q_seq_lens is not None) and (batch_valid_length is not None) and caches_exist:
            is_second_plus = self._is_second_plus_chunk(self._Cast(q_seq_lens, mstype.int32))
            kv_end = self._Add(self._Cast(batch_valid_length, mstype.int32),
                               self._Cast(q_seq_lens, mstype.int32))
            aligned_q  = self._all_x16(self._Cast(q_seq_lens, mstype.int32))
            aligned_kv = self._all_x16(kv_end)
            if is_second_plus and aligned_q and aligned_kv:
                # gather contiguous KV from cache (dtype = cache dtype)
                max_kv_len_py = int(kv_end.max().asnumpy())
                k_contig, v_contig = self._gather_contiguous_kv(
                    key_cache, value_cache,
                    self._Cast(block_tables, mstype.int32),
                    self._Cast(batch_valid_length, mstype.int32),
                    self._Cast(q_seq_lens, mstype.int32),
                    max_kv_len_py=max_kv_len_py
                )
                # For FA: q/k/v must share dtype. Cast q to k_contig dtype if needed.
                if query.dtype != k_contig.dtype:
                    query = self._Cast(query, k_contig.dtype)
                pm, am, aql, akl = self._sanitize_fa_masks(query, k_contig, padding_mask, attn_mask,
                                                           actual_seq_qlen, kv_end)
                self._dbg("FA 2nd+ chunk; dtype=", query.dtype, " kv_end=", akl)
                _, _, _, ctx = self.flash_attention(query, k_contig, v_contig, pm, am, aql, akl)
                return ctx

        # PA (decode or fallback)
        # Make sure PA mask matches target dtype and query matches cache dtype.
        if caches_exist and query.dtype != key_cache.dtype:
            query = self._Cast(query, key_cache.dtype)
        if attn_mask is not None and attn_mask.dtype != (key_cache.dtype if caches_exist else query.dtype):
            attn_mask = self._Cast(attn_mask, key_cache.dtype if caches_exist else query.dtype)

        self._dbg("PA decode/fallback; dtype=", query.dtype)
        return self._run_paged(query, key_cache, value_cache, block_tables,
                               batch_valid_length, q_seq_lens, attn_mask)

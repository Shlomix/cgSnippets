# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Flash Attention Layer"""
__all__ = ['FlashAttention']

import os
from mindspore import ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """Flash Attention Layer with optional FA on 2nd+ chunk prefill.

    first prefill    → FlashAttention(query, key, value)
    2nd+ chunk       → (optional) gather contiguous KV from paged cache → FlashAttention(query, Kc, Vc)
    decode / others  → PagedAttention

    Env:
      MF_FORCE_FA_CHUNK=1  → enable FA for 2nd+ chunk prefill (default: 1)
      MF_DEBUG_ATTENTION=1 → enable FA prints (default: 1)
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
        self.debug = os.getenv("MF_DEBUG_ATTENTION", "1") == "1"

        # core ops
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

        # utility ops
        self._print_op   = ops.Print()
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
        self._Sub        = ops.Sub()
        self._FloorDiv   = ops.FloorDiv()
        self._Mod        = ops.Mod()
        self._Gather     = ops.Gather()
        self._GatherD    = ops.GatherD()
        self._Fill       = ops.Fill()
        self._Select     = ops.Select()
        self._LogicalAnd = ops.LogicalAnd()
        self._LogicalNot = ops.LogicalNot()

    # ---------- tiny debug helper ----------
    def _dbg(self, *args):
        if self.debug:
            self._print_op(*args)

    # ---------- small helpers ----------
    def _scalar_bool(self, b):
        return ops.scalar_to_tensor(bool(b), mstype.bool_)

    def _int32(self, x):
        return self._Cast(x, mstype.int32)

    def _classify_step(self, q_seq_lens, actual_seq_qlen, actual_seq_kvlen, batch_valid_length, bsz):
        """Simplified, graph-safe classifier:
           first_prefill      := is_prefill
           second_plus_chunk  := (!is_prefill) && any(q_len > 1)
           is_decode_t        := !any(q_len > 1)  (informational)"""
        is_prefill_t = self._scalar_bool(self.is_prefill)
        if q_seq_lens is not None:
            has_chunk_t = self._Greater(self._ReduceMax(q_seq_lens), ops.scalar_to_tensor(1, q_seq_lens.dtype))
        else:
            has_chunk_t = self._scalar_bool(False)

        first_prefill_t     = is_prefill_t
        second_plus_chunk_t = self._LogicalAnd(self._LogicalNot(is_prefill_t), has_chunk_t)
        is_decode_t         = self._LogicalNot(has_chunk_t)

        self._dbg("FA.cls is_prefill=", is_prefill_t,
                  " has_chunk=", has_chunk_t,
                  " second_plus_chunk=", second_plus_chunk_t)
        return is_decode_t, first_prefill_t, second_plus_chunk_t

    def _gather_contiguous_kv(self, key_cache, value_cache, block_tables, batch_valid_length, q_seq_lens, bsz):
        """Gather contiguous KV of shape [B, S_kv, Hk*D] from paged cache. Graph-safe, int32 indices."""
        # casts
        block_tables       = self._int32(block_tables)
        batch_valid_length = self._int32(batch_valid_length)
        q_seq_lens         = self._int32(q_seq_lens)

        # compute kv end lengths and positions
        kv_end_len = self._Add(batch_valid_length, q_seq_lens)            # [B]
        max_kv_len = self._ReduceMax(kv_end_len)                           # scalar int32

        pos = self._Range(ops.scalar_to_tensor(0, mstype.int32),
                          max_kv_len,
                          ops.scalar_to_tensor(1, mstype.int32))          # [max_kv_len]
        pos_mat = self._Tile(self._ExpandDims(pos, 0), (bsz, 1))           # [B, max_kv_len]
        mask = self._Less(pos_mat, self._ExpandDims(kv_end_len, 1))        # [B, max_kv_len] bool

        BS = self._int32(ops.scalar_to_tensor(key_cache.shape[1], mstype.int32))
        blk_ids = self._FloorDiv(pos_mat, BS)                              # [B, max_kv_len]
        offs    = self._Mod(pos_mat,      BS)                              # [B, max_kv_len]

        phys_blk = self._GatherD(block_tables, 1, blk_ids)                 # [B, max_kv_len]
        global_idx = self._Add(self._Mul(phys_blk, BS), offs)              # [B, max_kv_len]
        flat_idx = self._Reshape(global_idx, (-1,))                        # [B*max_kv_len]

        kc_shape = key_cache.shape
        NB, BSv = kc_shape[0], kc_shape[1]

        if len(kc_shape) == 4:
            Hk, D = kc_shape[2], kc_shape[3]
            k_flat = self._Reshape(key_cache,   (NB * BSv, Hk * D))
            v_flat = self._Reshape(value_cache, (NB * BSv, Hk * D))
        else:
            HkD = kc_shape[2]
            k_flat = self._Reshape(key_cache,   (NB * BSv, HkD))
            v_flat = self._Reshape(value_cache, (NB * BSv, HkD))

        k_rows = self._Gather(k_flat, flat_idx, 0)                         # [B*max_kv_len, Hk*D]
        v_rows = self._Gather(v_flat, flat_idx, 0)
        k_full = self._Reshape(k_rows, (bsz, max_kv_len, -1))              # [B, max_kv_len, Hk*D]
        v_full = self._Reshape(v_rows, (bsz, max_kv_len, -1))

        # zero masked tail so FA ignores padding (it also receives lengths)
        mask_f = self._Cast(self._ExpandDims(mask, 2), k_full.dtype)       # [B, max_kv_len, 1]
        k_contig = self._Mul(k_full, mask_f)
        v_contig = self._Mul(v_full, mask_f)

        self._dbg("FA.gather k/v:", str(k_contig.shape), "/", str(v_contig.shape))
        return k_contig, v_contig

    # ------------------------- forward -------------------------
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
        # Update cache on every step (unless MLA)
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        bsz = query.shape[0]
        self._dbg("FA[enter] bsz=", str(bsz),
                  " q=", str(query.shape),
                  " k=", str(key.shape),
                  " v=", str(value.shape),
                  " is_prefill=", str(self.is_prefill),
                  " force_fa_chunk2p=", str(self.force_fa_chunk2p))

        # classify (tensor booleans)
        is_decode_t, first_prefill_t, second_plus_chunk_t = self._classify_step(
            q_seq_lens, actual_seq_qlen, actual_seq_kvlen, batch_valid_length, bsz
        )

        # ---- First prefill → FA(query, key, value) ----
        if self.is_prefill:
            self._dbg("FA path: first_prefill → FlashAttention")
            _, _, _, ctx_prefill = self.flash_attention(
                query, key, value,
                None, None,
                padding_mask, attn_mask,
                prefix, actual_seq_qlen, actual_seq_kvlen
            )
            return ctx_prefill

        # ---- Non-prefill: baseline PagedAttention ----
        if self.use_multi_latent_attention:
            paged_ctx = self.paged_attention(
                query, key_cache, key_cache,
                block_tables, batch_valid_length,
                None, None, attn_mask, q_seq_lens
            )
        else:
            paged_ctx = self.paged_attention(
                query, key_cache, value_cache,
                block_tables, batch_valid_length,
                None, None, attn_mask, q_seq_lens
            )

        # ---- Optional FA for 2nd+ chunk (contiguous KV + FA) ----
        if (not self.use_multi_latent_attention) and self.force_fa_chunk2p:
            self._dbg("FA considering 2nd+ chunk path")
            k_contig, v_contig = self._gather_contiguous_kv(
                key_cache, value_cache, block_tables, batch_valid_length, q_seq_lens, bsz
            )
            _, _, _, fa_ctx = self.flash_attention(
                query, k_contig, v_contig,
                None, None,
                padding_mask, attn_mask,
                prefix, actual_seq_qlen, actual_seq_kvlen
            )
            out_ctx = self._Select(second_plus_chunk_t, fa_ctx, paged_ctx)
            self._dbg("FA path: Select(2nd+ ? FA : PagedAttn)")
            return out_ctx

        # default non-prefill
        self._dbg("FA path: paged_attention (default)")
        return paged_ctx

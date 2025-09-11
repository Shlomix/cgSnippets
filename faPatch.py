# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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
from mindspore import ops
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """Flash Attention Layer with chunk-prefill support via contiguous KV gather.

    Args：
        - head_num (int): Number of attention heads.
        - head_dim (Optional[int]): Dimension of each attention head. Default: None.
        - kv_head_num (Optional[int]): Number of key-value heads. Default: None
        - keep_prob (float): Dropout keep probability. Default: 1.0.
        - scale_value (float): Scaling factor for attention scores. Default: 1.0.
        - pre_tokens (int): Number of previous tokens to consider. Default: 2147483647.
        - next_tokens (int): Number of next tokens to consider. Default: 2147483647.
        - sparse_mode (int): Mode for sparse attention. Default: 0.
        - input_layout (str): Layout of input tensors("TH" or "TND"). Default: "TH".
        - pa_kv_head_num (Optional[int]): Key-value head number for PagedAttention. Default: None.
        - pa_mla_v_dim (int): Dimension for multi-latent attention. Default: 0.

     Inputs:
        - **query** (Tensor[float16, bfloat16]) - [B, S_q, H_q]
        - **key** (Tensor[float16, bfloat16]) - [B, S_k, H_k] (first prefill path)
        - **value** (Tensor[float16, bfloat16]) - [B, S_k, H_k] (first prefill path)
        - **slot_mapping** (Tensor) - physical slots for new tokens.
        - **block_tables** (Tensor) - page table [B, max_blocks].
        - **batch_valid_length** (Tensor[int32]) - prefix length before this step [B].
        - **context_lens_tensor** (Tensor) - (unused here).
        - **q_seq_lens** (Tensor[int32]) - new tokens per sequence this step [B].
        - **actual_seq_qlen** (Tensor[int32]) - effective Q length per sequence [B].
        - **actual_seq_kvlen** (Tensor[int32]) - effective KV length per sequence [B].
        - **attn_mask** (Tensor or None)
        - **padding_mask** (None) - reserved.
        - **prefix** (Tensor[int64] or None) - reserved.
        - **key_cache** / **value_cache** - paged caches:
            shape either [NB, BS, Hk, D] or [NB, BS, Hk*D].

    Outputs:
        - **context** (Tensor) - same BSH as query.
    """

    def __init__(
            self,
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
            pa_mla_v_dim=0
    ):
        super().__init__()
        self.head_num = head_num
        self.hidden_size_per_attention_head = head_dim
        self.kv_head_num = kv_head_num
        self.sparse_mode = sparse_mode
        self.is_prefill = True
        self.input_layout = input_layout
        self.use_multi_latent_attention: bool = pa_mla_v_dim > 0

        # Toggle: force FlashAttention for second+ chunk prefill (default ON)
        self.force_fa_chunk2p = os.getenv("MF_FORCE_FA_CHUNK", "1") == "1"
        # Toggle: very lightweight debug prints (default ON)
        self.debug = os.getenv("MF_DEBUG_ATTENTION", "1") == "1"

        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.input_layout,
            sparse_mode=self.sparse_mode)

        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim)

        # ops used in graph
        self._Print      = ops.Print()
        self._Range      = ops.Range()
        self._Reshape    = ops.Reshape()
        self._ExpandDims = ops.ExpandDims()
        self._Tile       = ops.Tile()
        self._Cast       = ops.Cast()
        self._ReduceMax  = ops.ReduceMax(keep_dims=False)
        self._ReduceSum  = ops.ReduceSum(keep_dims=False)
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

    def _maybe_print(self, *args):
        if self.debug:
            self._Print(*args)

    def _classify_step(self, q_seq_lens, actual_seq_qlen, actual_seq_kvlen, batch_valid_length, bsz):
        """Return (is_decode, is_first_prefill, is_second_plus_chunk)."""
        is_decode = False
        is_first_prefill = False
        is_second_plus_chunk = False

        if q_seq_lens is not None:
            sum_q = self._ReduceSum(q_seq_lens)  # scalar
            is_decode = bool(self._Equal(sum_q, ops.scalar_to_tensor(bsz, sum_q.dtype)).asnumpy())

        kv_longer_than_q = False
        same_len = False
        if (actual_seq_qlen is not None) and (actual_seq_kvlen is not None):
            max_q  = self._ReduceMax(actual_seq_qlen)
            max_kv = self._ReduceMax(actual_seq_kvlen)
            kv_longer_than_q = bool(self._Greater(max_kv, max_q).asnumpy())
            same_len = bool(self._Equal(max_kv, max_q).asnumpy())

            self._maybe_print("FA.cls max_q=", str(max_q), " max_kv=", str(max_kv))
        else:
            self._maybe_print("FA.cls actual_seq_qlen or actual_seq_kvlen is None")

        # "first prefill" usually flagged by self.is_prefill and has same_len True (no longer prefix yet)
        is_first_prefill = bool(self.is_prefill) and same_len
        # 2d+ chunk when not first prefill, not decode, and kv prefix is longer than this step's q
        is_second_plus_chunk = (not bool(self.is_prefill)) and (not is_decode) and kv_longer_than_q

        self._maybe_print("FA.cls is_prefill=", str(self.is_prefill),
                          " is_decode=", str(is_decode),
                          " first_prefill=", str(is_first_prefill),
                          " second_plus_chunk=", str(is_second_plus_chunk))
        return is_decode, is_first_prefill, is_second_plus_chunk

    def _gather_contiguous_kv(self, key_cache, value_cache, block_tables, batch_valid_length, q_seq_lens, bsz):
        """Create a contiguous KV view [B, max_kv_len, Hk*D] from paged cache."""
        # kv_end_len[b] = prefix_len[b] + q_len_this_step[b]
        kv_end_len = self._Add(batch_valid_length, q_seq_lens)          # [B]
        max_kv_len = self._ReduceMax(kv_end_len)                         # scalar
        self._maybe_print("FA.gather max_kv_len=", str(max_kv_len))

        # positions per batch
        pos = self._Range(ops.scalar_to_tensor(0, ops.int32),
                          max_kv_len,
                          ops.scalar_to_tensor(1, ops.int32))            # [max_kv_len]
        pos_mat = self._Tile(self._ExpandDims(pos, 0), (bsz, 1))         # [B, max_kv_len]

        # valid mask
        mask = self._Less(pos_mat, self._ExpandDims(kv_end_len, 1))      # [B, max_kv_len]

        BS = value_cache.shape[1]                                        # block size
        blk_ids = self._FloorDiv(pos_mat, ops.scalar_to_tensor(BS, ops.int32))  # [B, max_kv_len]
        offs    = self._Mod(pos_mat,      ops.scalar_to_tensor(BS, ops.int32))  # [B, max_kv_len]

        # physical block IDs per (b, pos)
        phys_blk = self._GatherD(block_tables, 1, blk_ids)               # [B, max_kv_len]

        # flatten index into [NB*BS, ...]
        global_idx = self._Add(self._Mul(phys_blk, ops.scalar_to_tensor(BS, ops.int32)), offs)  # [B, max_kv_len]
        flat_idx   = self._Reshape(global_idx, (-1,))                    # [B*max_kv_len]

        kc_shape = key_cache.shape
        NB = kc_shape[0]

        if len(kc_shape) == 4:
            Hk, D = kc_shape[2], kc_shape[3]
            k_flat = self._Reshape(key_cache,   (NB * BS, Hk * D))       # [NB*BS, Hk*D]
            v_flat = self._Reshape(value_cache, (NB * BS, Hk * D))       # [NB*BS, Hk*D]
        else:
            # [NB, BS, Hk*D] → [NB*BS, Hk*D]
            HkD = kc_shape[2]
            k_flat = self._Reshape(key_cache,   (NB * BS, HkD))
            v_flat = self._Reshape(value_cache, (NB * BS, HkD))

        # gather and reshape back to [B, max_kv_len, Hk*D]
        k_rows = self._Gather(k_flat, flat_idx, 0)                       # [B*max_kv_len, Hk*D]
        v_rows = self._Gather(v_flat, flat_idx, 0)
        k_full = self._Reshape(k_rows, (bsz, max_kv_len, -1))
        v_full = self._Reshape(v_rows, (bsz, max_kv_len, -1))

        # zero out padded tail
        mask_f  = self._Cast(self._ExpandDims(mask, 2), k_full.dtype)    # [B, max_kv_len, 1]
        k_contig = self._Mul(k_full, mask_f)
        v_contig = self._Mul(v_full, mask_f)

        self._maybe_print("FA.gather k_contig=", str(k_contig.shape),
                          " v_contig=", str(v_contig.shape))
        return k_contig, v_contig

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
        # Always reshape & write new tokens into cache when not MLA.
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        bsz = query.shape[0]
        self._maybe_print("FA[enter] bsz=", str(bsz),
                          " q=", str(query.shape),
                          " k=", str(key.shape),
                          " v=", str(value.shape),
                          " use_mla=", str(self.use_multi_latent_attention),
                          " is_prefill=", str(self.is_prefill),
                          " force_fa_chunk2p=", str(self.force_fa_chunk2p))

        # Classify step (decode / first prefill / 2d+ chunk prefill)
        is_decode, is_first_prefill, is_second_plus_chunk = self._classify_step(
            q_seq_lens, actual_seq_qlen, actual_seq_kvlen, batch_valid_length, bsz
        )

        # --- First prefill: use FA directly on (query, key, value) ---
        if is_first_prefill:
            self._maybe_print("FA path = first_prefill → FlashAttention")
            _, _, _, context = self.flash_attention(
                query, key, value,
                None, None,
                padding_mask, attn_mask,
                prefix, actual_seq_qlen, actual_seq_kvlen
            )
            return context

        # --- Second+ chunk prefill: optionally force FA with contiguous KV ---
        if (not self.use_multi_latent_attention) and self.force_fa_chunk2p and is_second_plus_chunk:
            self._maybe_print("FA path = second_plus_chunk → gather(KV) + FlashAttention")
            k_contig, v_contig = self._gather_contiguous_kv(
                key_cache, value_cache, block_tables, batch_valid_length, q_seq_lens, bsz
            )
            # Call FA on (query, k_contig, v_contig)
            _, _, _, context = self.flash_attention(
                query, k_contig, v_contig,
                None, None,
                padding_mask, attn_mask,
                prefix, actual_seq_qlen, actual_seq_kvlen
            )
            return context

        # --- Decode or fallbacks → PagedAttention ---
        if self.use_multi_latent_attention:
            self._maybe_print("FA path = paged_attention (MLA)")
            context = self.paged_attention(
                query, key_cache, key_cache,
                block_tables, batch_valid_length, None,
                None, attn_mask, q_seq_lens
            )
        else:
            self._maybe_print("FA path = paged_attention (default)")
            context = self.paged_attention(
                query, key_cache, value_cache,
                block_tables, batch_valid_length, None,
                None, attn_mask, q_seq_lens
            )
        return context

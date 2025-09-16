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

import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """Flash Attention Layer.

    This function contains the flash attention primitives used in FlashAttention (see paper) and PagedAttention.
    `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/pdf/2205.14135.pdf>`

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
        - **query** (Tensor[float16, bfloat16]) - The query tensor.
        - **key** (Tensor[float16, bfloat16]) - The key tensor.
        - **value** (Tensor[float16, bfloat16]) - The value tensor.
        - **slot_mapping** (Tensor) - Store token cache physical slot index.
        - **block_tables** (Tensor) - The block mapping table with data type of int32.
        - **batch_valid_length** (Tensor) - In incremental inference, a tensor used for calculating the index
          of the previous step. It is of int32 type and has a shape of [batch_size].
        - **context_lens_tensor** (Tensor) - The context length of each sequence with data type of int32.
        - **q_seq_lens** (Tensor) - Query sequence lengths for PagedAttention.
        - **actual_seq_qlen** (Union[List[int64], Tuple[int64], None]) - Size of query corresponding to each batch,
          array with increasing values and the last value equal to T1.
        - **actual_seq_kvlen** (Union[List[int64], Tuple[int64], None]) - Size of key and value corresponding to
          each batch, array with increasing values and the last value equal to T2.
        - **attn_mask** (Union[Tensor, None]) - The attention mask tensor.
        - **padding_mask** (None) - Reserved parameter. Not implemented yet.
        - **prefix** (Union[Tensor[int64], None]) - N value of each Batch in the prefix sparse calculation scenario.
          Not implemented yet. Input tensor of shape :math:`(B,)`.
        - **key_cache** (Tensor, optional) - Key cache for incremental inference.
        - **value_cache** (Tensor, optional) - Value cache for incremental inference.

    Outputs:
        - **context** (Tensor[float16, bfloat16]) - The output of flash attention. its shape, and data type
          are the same as the query.

    Supported Platforms:
        ``Ascend``
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

        # --- Core ops ---
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        # Original FA (TH layout path, unchanged)
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.input_layout,
            sparse_mode=self.sparse_mode)
        self.flash_attention.add_prim_attr("mf_role", "fa_prefill_TH")

        # Additional FA for 2D(+)-chunk prefill (TND layout; causal; next_tokens=0)
        self._flash_attention_tnd = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,   # already INT_MAX by default; satisfies pre_tokens >= max_q_len
            next_tokens=0,
            inner_precise=0,
            input_layout="TND",
            sparse_mode=3)           # causal optimized
        self._flash_attention_tnd.add_prim_attr("mf_role", "fa_chunk_TND")

        # Decode kernel (unchanged)
        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim)

        # --- helper kernels ---
        self._reshape = ops.Reshape()
        self._cast = ops.Cast()
        self._tile = ops.Tile()
        self._expand_dims = ops.ExpandDims()
        self._reduce_max = ops.ReduceMax()
        self._range = ops.Range()
        self._greater = ops.Greater()
        self._gather = ops.Gather()      # (params, indices, axis)
        self._gather_d = ops.GatherD()   # gather_d(x, dim, index)
        self._masked_select = ops.MaskedSelect()
        self._concat0 = ops.Concat(axis=0)

        # Canonical (2048,2048) lower-tri discard mask (uint8) for TND mode (built lazily)
        self._tnd_mask_2048 = None

    # -------------------- helpers --------------------

    def _has_tnd_ragged(self, actual_seq_qlen, actual_seq_kvlen):
        # TND requires both cumulative arrays; avoid relying on is_prefill flag.
        if actual_seq_qlen is None or actual_seq_kvlen is None:
            return False
        if len(actual_seq_qlen.shape) == 0 or len(actual_seq_kvlen.shape) == 0:
            return False
        return True

    def _infer_head_dim(self, query):
        if self.hidden_size_per_attention_head is not None:
            return self.hidden_size_per_attention_head
        # fallback inference from packed TH: (T, H*D)
        return query.shape[1] // self.head_num

    def _to_tnd_from_th(self, x, n_heads, head_dim):
        # (T, H*D) -> (T, N, D)
        if x is None:
            return None
        t, hd = x.shape
        if hd == n_heads * head_dim:
            return self._reshape(x, (t, n_heads, head_dim))
        return x

    def _from_tnd_to_th(self, x):
        # (T, N, D) -> (T, N*D)
        if x is None:
            return None
        if len(x.shape) == 3:
            t, n, d = x.shape
            return self._reshape(x, (t, n * d))
        return x

    def _diff_lengths(self, cum_lengths):
        # cumulative [l1, l1+l2, ...] -> per-seq [l1, l2, ...]; cum_lengths shape [B]
        B = cum_lengths.shape[0]
        zero = self._cast(self._range(0, 1, 1), cum_lengths.dtype)  # (1,)
        prev = self._concat0((zero, cum_lengths[:B-1]))
        return cum_lengths - prev

    def _kv_from_cache_tnd(self, cache, block_tables, actual_seq_kvlen):
        """
        Build contiguous (T2, N_kv, D) from block-wise cache by using block_tables and ragged kv lengths.

        cache:        (num_blocks, block_size, N_kv, D)
        block_tables: (B, max_blocks_per_seq)  int32
        actual_seq_kvlen: cumulative kv lengths [B], last == T2
        """
        nb, bs, n_kv, d = cache.shape
        flat = self._reshape(cache, (nb * bs, n_kv, d))  # (nb*bs, N_kv, D)

        kv_cum = actual_seq_kvlen
        kv_lens = self._diff_lengths(kv_cum)                  # [B]
        max_len = self._reduce_max(kv_lens)                   # scalar
        max_len_i32 = self._cast(max_len, mstype.int32)

        pos = self._range(self._cast(0, mstype.int32), max_len_i32, self._cast(1, mstype.int32))  # (max_len,)
        B = kv_lens.shape[0]
        pos = self._tile(self._expand_dims(pos, 0), (B, 1))   # [B, max_len]

        kv_lens_i32 = self._cast(kv_lens, mstype.int32)
        kv_lens_2d = self._expand_dims(kv_lens_i32, 1)        # [B,1]
        valid_mask = self._greater(kv_lens_2d, pos)           # True where pos < len

        blk_idx = pos // bs                                   # [B, max_len]
        table_i32 = self._cast(block_tables, mstype.int32)    # [B, max_blocks]
        blk_ids = self._gather_d(table_i32, 1, blk_idx)       # [B, max_len]
        offsets = pos - (blk_idx * bs)                        # [B, max_len]
        global_idx = blk_ids * bs + offsets                   # [B, max_len]

        valid_idx_flat = self._masked_select(global_idx, valid_mask)  # [T2]
        kv_tnd = self._gather(flat, valid_idx_flat, 0)                # (T2, N_kv, D)
        return kv_tnd

    def _ensure_tnd_causal_mask(self, attn_mask, size=2048):
        """
        TND requires a lower-triangular mask of shape (2048, 2048),
        dtype bool/uint8, with 1=discard and 0=keep. If incoming mask doesn't
        match, build the canonical one.
        """
        if self._tnd_mask_2048 is None:
            size_i32 = self._cast(size, mstype.int32)
            idx = self._range(self._cast(0, mstype.int32), size_i32, self._cast(1, mstype.int32))  # (size,)
            row = self._tile(self._expand_dims(idx, 1), (size_i32, 1))  # (size, size)
            col = self._tile(self._expand_dims(idx, 0), (1, size_i32))  # (size, size)
            upper = self._greater(col, row)  # True on strictly upper triangle (j > i)
            self._tnd_mask_2048 = self._cast(upper, mstype.uint8)  # 1=discard, 0=keep

        if attn_mask is not None:
            if len(attn_mask.shape) == 2 and attn_mask.shape[0] == 2048 and attn_mask.shape[1] == 2048:
                if attn_mask.dtype == mstype.uint8 or attn_mask.dtype == mstype.bool_:
                    return attn_mask
        return self._tnd_mask_2048

    # -------------------- forward --------------------

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
        # Keep the original cache write behavior (some pipelines flip is_prefill during chunk prefill).
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # 2D(+)-chunk prefill fast-path (independent of is_prefill flag):
        # Trigger when ragged TND descriptors are present; use K/V from cache.
        if self._has_tnd_ragged(actual_seq_qlen, actual_seq_kvlen):
            head_dim = self._infer_head_dim(query)
            q_tnd = self._to_tnd_from_th(query, self.head_num, head_dim)
            k_tnd = self._kv_from_cache_tnd(key_cache, block_tables, actual_seq_kvlen)
            v_tnd = self._kv_from_cache_tnd(value_cache, block_tables, actual_seq_kvlen)
            attn_mask_tnd = self._ensure_tnd_causal_mask(attn_mask)

            # Call TND FA; prefix must be None; next_tokens fixed in primitive.
            _, _, _, context = self._flash_attention_tnd(
                q_tnd, k_tnd, v_tnd,
                None, None,
                padding_mask, attn_mask_tnd,
                None,                      # prefix not used in TND
                actual_seq_qlen, actual_seq_kvlen
            )
            context = self._from_tnd_to_th(context)

        else:
            # Original behavior:
            if self.is_prefill:
                _, _, _, context = self.flash_attention(
                    query, key, value,
                    None, None,
                    padding_mask, attn_mask,
                    prefix, actual_seq_qlen,
                    actual_seq_kvlen)
            else:
                if self.use_multi_latent_attention:
                    context = self.paged_attention(query, key_cache, key_cache,
                                                   block_tables, batch_valid_length, None,
                                                   None, attn_mask, q_seq_lens)
                else:
                    context = self.paged_attention(query, key_cache, value_cache,
                                                   block_tables, batch_valid_length, None,
                                                   None, attn_mask, q_seq_lens)

        return context

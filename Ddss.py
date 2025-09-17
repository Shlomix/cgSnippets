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

from mindspore import ops
import mindspore.common.dtype as mstype
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

    # ----------------- helpers for chunk prefill (TH) -----------------

    def _diff_lengths(self, cum_lengths):
        """cum [l1, l1+l2, ...] -> [l1, l2, ...] (keeps dtype; no casts on Parameters)."""
        B = cum_lengths.shape[0]
        zeros = ops.Zeros()((1,), cum_lengths.dtype)
        prev = ops.Concat(axis=0)((zeros, cum_lengths[:B - 1])) if B > 1 else zeros
        return cum_lengths - prev

    def _kv_cache_to_th_from_paged(self, cache, block_tables, actual_seq_kvlen):
        """
        Convert paged cache (num_blocks, block_size, N_kv, D) into TH (Tk_total, N_kv*D)
        in logical time order per sequence using block_tables + actual_seq_kvlen.
        """
        # Local ops
        reshape = ops.Reshape()
        cast = ops.Cast()
        expand = ops.ExpandDims()
        rng = ops.Range()
        reduce_max = ops.ReduceMax()
        greater = ops.Greater()
        gather = ops.Gather()
        gather_d = ops.GatherD()
        masked_select = ops.MaskedSelect()
        concat0 = ops.Concat(axis=0)
        zeros = ops.Zeros()

        nb, bs, n_kv, d = cache.shape
        flat = reshape(cache, (nb * bs, n_kv, d))  # (nb*bs, N_kv, D)

        kv_lens = self._diff_lengths(actual_seq_kvlen)  # (B,)
        B = kv_lens.shape[0]

        # positions [0..max_len-1] (int32 synthetic)
        max_len = reduce_max(kv_lens)
        max_len_i32 = cast(max_len, mstype.int32)
        pos_i32 = rng(ops.scalar_to_tensor(0, mstype.int32),
                      max_len_i32,
                      ops.scalar_to_tensor(1, mstype.int32))  # (L,)

        # validity mask via broadcasting (no Tile)
        pos_for_mask = cast(expand(pos_i32, 0), kv_lens.dtype)  # (1, L)
        kv_lens_2d = expand(kv_lens, 1)                         # (B, 1)
        valid_mask = greater(kv_lens_2d, pos_for_mask)          # (B, L) bool

        # block indexing
        bs_i32 = ops.scalar_to_tensor(bs, mstype.int32)
        blk_idx_row = pos_i32 // bs_i32                         # (L,)
        zeros_B1 = zeros((B, 1), mstype.int32)
        blk_idx = zeros_B1 + blk_idx_row                        # (B, L)

        table_i32 = cast(block_tables, mstype.int32)            # (B, max_blocks)
        blk_ids = gather_d(table_i32, 1, blk_idx)               # (B, L)
        offsets = pos_i32 - (blk_idx_row * bs_i32)              # (L,)
        offsets = zeros_B1 + offsets                            # (B, L)
        global_idx = blk_ids * bs_i32 + offsets                 # (B, L)

        # ragged pack: flatten valid positions into time-major index vector
        valid_idx_flat = masked_select(global_idx, valid_mask)  # (Tk_total,)
        kv_tnd = gather(flat, valid_idx_flat, 0)                # (Tk_total, N_kv, D)

        # reshape to TH for FA(TH)
        Tk_total = kv_tnd.shape[0]
        kv_th = reshape(kv_tnd, (Tk_total, n_kv * d))           # (Tk_total, N_kv*D)
        return kv_th

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

        # --- Minimal safety: ReshapeAndCache expects slot_mapping int32. Keep everything else as-is. ---
        slot_for_cache = slot_mapping
        if slot_mapping is not None and slot_mapping.dtype != mstype.int32:
            slot_for_cache = ops.Cast()(slot_mapping, mstype.int32)

        # Always write fresh K/V into cache (original behavior).
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_for_cache)

        # -------- Branching (minimal change) --------
        # Original prefill path (TH FA with provided key/value)
        if self.is_prefill:
            _, _, _, context = self.flash_attention(query, key, value,
                                                    None, None,
                                                    padding_mask, attn_mask,
                                                    prefix, actual_seq_qlen,
                                                    actual_seq_kvlen)
            return context

        # 2D+ chunk prefill detector: any q_seq_lens > 1 and not prefill
        is_chunk_prefill = False
        if q_seq_lens is not None:
            # Avoid importing extra ops: simple reduce-max > 1
            reduce_max = ops.ReduceMax()
            max_q = reduce_max(q_seq_lens)
            # max_q is a scalar tensor; compare to 1 in tensor domain
            is_chunk_prefill = ops.Greater()(max_q, ops.scalar_to_tensor(1, max_q.dtype))

        # If chunk prefill -> build TH K/V from cache and call TH FA
        if is_chunk_prefill:
            # Build contiguous TH K/V in logical time order from paged cache
            k_th = self._kv_cache_to_th_from_paged(key_cache, block_tables, actual_seq_kvlen)
            v_th = self._kv_cache_to_th_from_paged(value_cache, block_tables, actual_seq_kvlen)

            # Call the *same* TH FlashAttention with (Q, K_th, V_th) and the ragged descriptors
            _, _, _, context = self.flash_attention(query, k_th, v_th,
                                                    None, None,
                                                    padding_mask, attn_mask,
                                                    prefix, actual_seq_qlen,
                                                    actual_seq_kvlen)
            return context

        # Fallback: decode path (unchanged)
        if self.use_multi_latent_attention:
            context = self.paged_attention(query, key_cache, key_cache,
                                           block_tables, batch_valid_length, None,
                                           None, attn_mask, q_seq_lens)
        else:
            context = self.paged_attention(query, key_cache, value_cache,
                                           block_tables, batch_valid_length, None,
                                           None, attn_mask, q_seq_lens)

        return context

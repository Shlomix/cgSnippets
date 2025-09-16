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

        # Remember requested scale; kernels get scale=1.0 to avoid BF16/double attr issues.
        self._external_scale = scale_value

        # --- Core ops (unchanged) ---
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        # Original FA (TH layout, unchanged path) — neutralize kernel scaling
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=1.0,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.input_layout,
            sparse_mode=self.sparse_mode)
        self.flash_attention.add_prim_attr("mf_role", "fa_prefill_TH")

        # Extra FA for 2D(+)-chunk prefill (TND layout; causal; next_tokens=0) — neutralize kernel scaling
        self._flash_attention_tnd = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=1.0,
            pre_tokens=pre_tokens,   # large enough by default
            next_tokens=0,
            inner_precise=0,
            input_layout="TND",
            sparse_mode=3)
        self._flash_attention_tnd.add_prim_attr("mf_role", "fa_chunk_TND")

        # Decode kernel — neutralize kernel scaling
        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, 1.0, pa_kv_head_num, mla_v_dim=pa_mla_v_dim)

        # --- helpers (Tile-free; never cast Parameters) ---
        self._reshape = ops.Reshape()
        self._cast = ops.Cast()
        self._expand_dims = ops.ExpandDims()
        self._range = ops.Range()
        self._reduce_max = ops.ReduceMax()
        self._greater = ops.Greater()
        self._gather = ops.Gather()
        self._gather_d = ops.GatherD()
        self._masked_select = ops.MaskedSelect()
        self._concat0 = ops.Concat(axis=0)
        self._zeros = ops.Zeros()
        self._ones = ops.Ones()
        self._triu_strict = ops.Triu(diagonal=1)  # strictly upper triangle (j > i)

        # Canonical (2048,2048) lower-tri discard mask for TND mode
        self._tnd_mask_2048 = None

    # -------------------- helpers --------------------

    def _has_tnd_ragged(self, actual_seq_qlen, actual_seq_kvlen):
        # TND requires both cumulative arrays; check rank >= 1.
        if actual_seq_qlen is None or actual_seq_kvlen is None:
            return False
        if len(actual_seq_qlen.shape) == 0 or len(actual_seq_kvlen.shape) == 0:
            return False
        return True

    def _infer_head_dim(self, query):
        if self.hidden_size_per_attention_head is not None:
            return self.hidden_size_per_attention_head
        return query.shape[1] // self.head_num  # packed TH: (T, H*D)

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
        # cumulative [l1, l1+l2, ...] -> per-seq [l1, l2, ...]; cum_lengths: [B]
        B = cum_lengths.shape[0]
        zero = self._zeros((1,), cum_lengths.dtype)                  # same dtype as input (Parameter-safe)
        prev = self._concat0((zero, cum_lengths[:B - 1]))            # (B,)
        return cum_lengths - prev                                    # (B,)

    def _kv_from_cache_tnd(self, cache, block_tables, actual_seq_kvlen):
        """
        Build contiguous (T2, N_kv, D) from block-wise cache using block_tables and ragged kv lengths.

        cache:        (num_blocks, block_size, N_kv, D)
        block_tables: (B, max_blocks_per_seq) int32
        actual_seq_kvlen: cumulative kv lengths [B], last == T2
        """
        nb, bs, n_kv, d = cache.shape
        flat = self._reshape(cache, (nb * bs, n_kv, d))              # (nb*bs, N_kv, D)

        kv_cum  = actual_seq_kvlen                                   # keep original dtype (may be Parameter[int])
        kv_lens = self._diff_lengths(kv_cum)                         # (B,) same dtype as input
        B       = kv_lens.shape[0]

        # Build positions [0..max_len-1] as int32 synthetic tensor
        max_len     = self._reduce_max(kv_lens)                      # scalar (same dtype as kv_lens)
        max_len_i32 = self._cast(max_len, mstype.int32)              # cast only the result (Tensor, not Parameter)

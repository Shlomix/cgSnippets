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
from mindspore import ops
import mindspore.common.dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """Flash Attention Layer.

    This function contains the flash attention primitives used in FlashAttention (see paper) and PagedAttention.
    `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/pdf/2205.14135.pdf>`
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

        # Optional override: force running FA for 2D(+)-chunk prefill path
        self.force_chunk_fa = os.getenv("MF_FORCE_CHUNK_FA", "").lower() in ("1", "true", "yes", "on")

        # Prims
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

        # Small helpers we use in the chunk path (MindSpore graph friendly)
        self._reshape = ops.Reshape()
        self._gather = ops.Gather()
        self._reduce_max = ops.ReduceMax()
        self._greater = ops.Greater()

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
        # Keep original behavior: always update KV cache
        if not self.use_multi_latent_attention:
            # NOTE: by request, no casting here (assumes caller/caches are consistent)
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # Prefill path: unchanged TH FA with provided key/value
        if self.is_prefill:
            _, _, _, context = self.flash_attention(query, key, value,
                                                    None, None,
                                                    padding_mask, attn_mask,
                                                    prefix, actual_seq_qlen,
                                                    actual_seq_kvlen)
            return context

        # 2D(+)-chunk prefill detector: not prefill AND any q_seq_lens > 1, or forced via env
        is_chunk = False
        if q_seq_lens is not None:
            max_q = self._reduce_max(q_seq_lens)
            is_chunk = self._greater(max_q, ops.scalar_to_tensor(1, max_q.dtype))
        if self.force_chunk_fa:
            # Force the branch regardless of runtime q_seq_lens
            is_chunk = True

        if is_chunk:
            # Prepare TH K/V directly from paged cache, per colleague's simple recipe
            # idx = block_tables.reshape(-1)
            idx = self._reshape(block_tables, (-1,))

            # Gather blocks along dim 0, then flatten the time axis and keep last-dim equal to original key/value's
            key_lin = self._gather(key_cache, idx, 0)     # (..., n_kv, d) with time packed by idx
            val_lin = self._gather(value_cache, idx, 0)

            # Reshape to TH: (-1, key.shape[1]) / (-1, value.shape[1]) — keep H*D width from incoming tensors
            key_th = self._reshape(key_lin, (-1, key.shape[1]))
            val_th = self._reshape(val_lin, (-1, value.shape[1]))

            # Call the same TH FlashAttention with rebuilt K/V and the ragged descriptors
            _, _, _, context = self.flash_attention(query, key_th, val_th,
                                                    None, None,
                                                    padding_mask, attn_mask,
                                                    prefix, actual_seq_qlen,
                                                    actual_seq_kvlen)
            return context

        # Decode path: unchanged (PagedAttention)
        if self.use_multi_latent_attention:
            context = self.paged_attention(query, key_cache, key_cache,
                                           block_tables, batch_valid_length, None,
                                           None, attn_mask, q_seq_lens)
        else:
            context = self.paged_attention(query, key_cache, value_cache,
                                           block_tables, batch_valid_length, None,
                                           None, attn_mask, q_seq_lens)

        return context

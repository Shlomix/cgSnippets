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
"""Flash Attention Layer with 2D+ Chunk Prefill from KV cache (TND)."""
__all__ = ["FlashAttention"]

import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """
    Flash Attention layer used in:
      • non-chunk prefill: FlashAttention (original TH layout path)
      • 2D(+)-chunk prefill: FlashAttention (TND layout, K/V from cache)
      • decode: PagedAttention

    TND rules (MindSpore):
      - input_layout="TND"
      - pass actual_seq_qlen and actual_seq_kvlen (cumulative; last == T)
      - causal sparse_mode (2 or 3); next_tokens=0; prefix=None
      - attn_mask must be (2048, 2048) lower-tri with 1=discard, 0=keep
    """

    def __init__(self,
                 head_num: int,
                 kv_head_num: int,
                 hidden_size_per_attention_head: int,
                 keep_prob: float,
                 scale_value: float,
                 pre_tokens: int,
                 next_tokens: int,
                 input_layout: str = "TH",
                 sparse_mode: int = 0,
                 pa_kv_head_num: int = None,
                 pa_mla_v_dim: int = 0):
        super().__init__()
        # Model/head config
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.hidden_size_per_attention_head = hidden_size_per_attention_head

        # Runt

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
"""Flash Attention Layer (TND-only for prefill & chunk prefill)"""
__all__ = ['FlashAttention']

import os
import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """Flash Attention wrapper with TND FA for both prefill and 2D(+)-chunk prefill, plus PagedAttention decode.

    Public API (ctor + construct signature) matches upstream.
    Casting of activations (e.g., to FP16) is handled by the caller.
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
            input_layout="TH",   # accepted but ignored; we run TND internally
            pa_kv_head_num=None,
            pa_mla_v_dim=0
    ):
        super().__init__()
        self.head_num = head_num
        self.hidden_size_per_attention_head = head_dim
        self.kv_head_num = kv_head_num
        self.sparse_mode = sparse_mode
        self.is_prefill = True
        self.input_layout = "TND"   # force TND internally
        self.use_multi_latent_attention: bool = pa_mla_v_dim > 0

        # ---- debug (env-gated) ----
        dbg_env = os.getenv("MF_FA_DEBUG", "").lower()
        self._debug = dbg_env not in ("", "0", "false", "off", "no")
        self._print = ops.Print()

        # --- Core ops ---
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        # TND FA for **prefill** (use caller's next_tokens/sparse_mode/scale_value)
        self.flash_attention_tnd_prefill = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout="TND",
            sparse_mode=self.sparse_mode)
        self.flash_attention_tnd_prefill.add_prim_attr("mf_role", "fa_TND_prefill")

        # TND FA for **2D(+)-chunk prefill** (strict causal; no right lookahead)
        self.flash_attention_tnd_chunk = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=0,          # strict causal for chunked prefill
            inner_precise=0,
            input_layout="TND",
            sparse_mo_

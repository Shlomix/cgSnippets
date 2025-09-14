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
from mindspore import ops, Tensor
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """
    Minimal FlashAttention+PagedAttention bridge used by parallel_core.Attention.

    Changes vs stock:
      * Always drive FlashAttention in **TND** layout (3-D) to avoid TH/2D tiling issues.
      * Force padding_mask=None for FlashAttentionScore (kernel requirement).
      * Keep decode path on PagedAttention as-is.
      * Return 2-D [T, H] so the caller (linear_proj) continues to work unchanged.
      * Lightweight debug prints gated by MF_DEBUG_FA=1.
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
            input_layout="TH",           # ignored, we enforce TND for FA
            pa_kv_head_num=None,
            pa_mla_v_dim=0
    ):
        super().__init__()
        # model params
        self.head_num = int(head_num)
        self.hidden_size_per_attention_head = int(head_dim) if head_dim is not None else None
        self.kv_head_num = kv_head_num
        self.sparse_mode = sparse_mode
        self.is_prefill = True
        self.use_multi_latent_attention: bool = pa_mla_v_dim > 0

        # tiny helpers
        self._reshape = P.Reshape()
        self._cast = P.Cast()
        self._dbg_on = os.getenv("MF_DEBUG_FA", "0") == "1"

        # kernels
        # NOTE: we **force** TND here; we will feed 3-D [T, N, D]
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout="TND",
            sparse_mode=self.sparse_mode
        )
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()
        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim
        )

    # ---------- small local utils ----------
    def _dbg(self, *msg):
        if self._dbg_on:
            print("[FA]", *msg, flush=True)

    def _ensure_same_dtype(self, q, k, v):
        # unify to q.dtype (we assume caches, if present, are fp16 as per your setup)
        dtype = q.dtype
        if k.dtype != dtype:
            k = self._cast(k, dtype)
        if v.dtype != dtype:
            v = self._cast(v, dtype)
        return q, k, v

    def _to_tnd(self, x, name):
        """
        Convert [B, S, H] or [T, H] -> [T, N, D] where H == N*D.
        """
        n = self.head_num
        d = self.hidden_size_per_attention_head
        if d is None:
            raise ValueError("FlashAttention: head_dim was None; please pass head_dim in constructor.")
        shape = x.shape
        if len(shape) == 3:
            b, s, h = shape
            # avoid exceptions in graph; rely on reshape to error if mismatch
            x = self._reshape(x, (b * s, n, d))
            self._dbg(f"{name}: BSH({b},{s},{h}) -> TND({b*s},{n},{d})")
        elif len(shape) == 2:
            t, h = shape
            x = self._reshape(x, (t, n, d))
            self._dbg(f"{name}: TH({t},{h}) -> TND({t},{n},{d})")
        else:
            self._dbg(f"{name}: unexpected rank {len(shape)} with shape {shape}")
            # Best effort: let runtime raise a clear reshape error.
            x = self._reshape(x, (-1, n, d))
        return x

    def _from_tnd_to_th(self, x_tnd):
        """
        Convert [T, N, D] -> [T, H] where H = N*D.
        """
        t, n, d = x_tnd.shape
        out = self._reshape(x_tnd, (t, n * d))
        self._dbg(f"OUT: TND({t},{n},{d}) -> TH({t},{n*d})")
        return out
    # --------------------------------------

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

        # Make types match; FA and PA both require consistent dtypes
        query, key, value = self._ensure_same_dtype(query, key, value)

        # Cache write (for decode) — only when caches exist; passing None will crash ReshapeAndCache
        if (not self.use_multi_latent_attention) and \
           (key_cache is not None) and (value_cache is not None) and (slot_mapping is not None):
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # Prefill path (first chunk and 2D+ chunks): use FlashAttention in **TND**
        if self.is_prefill:
            # Kernel requires padding_mask=None. Keep user attn_mask as-is.
            padding_mask = None

            # Convert inputs to TND
            q_tnd = self._to_tnd(query, "Q")
            k_tnd = self._to_tnd(key,   "K")
            v_tnd = self._to_tnd(value, "V")

            # Simple indicator for 2D+ chunk prefill (optional print)
            if self._dbg_on:
                # NOTE: these are Tensors in graph; printing shows shape/dtype/value head
                print("FA prefill: q_seq_lens=", q_seq_lens, " actual_seq_qlen=",
                      actual_seq_qlen, " actual_seq_kvlen=", actual_seq_kvlen, flush=True)

            # Call FA (TND) → returns TND
            # API: (q, k, v, real_shift, attn_bias, padding_mask, attn_mask, prefix, act_q, act_kv)
            _, _, _, context_tnd = self.flash_attention(
                q_tnd, k_tnd, v_tnd,
                None, None,
                padding_mask, attn_mask,
                prefix, actual_seq_qlen, actual_seq_kvlen
            )

            # Bring back to TH [T, H] for the caller (linear proj works on last dim)
            context = self._from_tnd_to_th(context_tnd)
            return context

        # Decode path: keep PagedAttention behavior untouched
        if self.use_multi_latent_attention:
            # (query, key_cache, key_cache, ...)
            context = self.paged_attention(query, key_cache, key_cache,
                                           block_tables, batch_valid_length,
                                           None, None, attn_mask, q_seq_lens)
        else:
            # (query, key_cache, value_cache, ...)
            context = self.paged_attention(query, key_cache, value_cache,
                                           block_tables, batch_valid_length,
                                           None, None, attn_mask, q_seq_lens)
        return context

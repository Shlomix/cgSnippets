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
"""Flash Attention Layer with 2D+ Chunk Prefill fast path."""
__all__ = ["FlashAttention"]

from mindspore import ops
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """
    Flash Attention layer used in prefill (FlashAttention) and decode (PagedAttention).

    This implementation keeps the original prefill path for packed TH layout and
    adds a 2D+ chunk-prefill fast path that switches to a TND-configured
    FlashAttentionScore when ragged descriptors are provided.

    Notes
    -----
    * Outside this module, Q/K/V are packed as (T, H*D); we reshape to (T, N, D)
      only for the TND fast path and then reshape back so surrounding code remains unchanged.
    * For chunked prefill we use causal optimized sparse mode with next_tokens=0.
    * GQA/MQA: Q is reshaped with head_num (Nq), K/V with kv_head_num (Nk).
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

        # Runtime flags (kept compatible with original code)
        self.sparse_mode = sparse_mode
        self.is_prefill = True            # toggled by the outer Attention module
        self.input_layout = input_layout
        self.use_multi_latent_attention: bool = (pa_mla_v_dim > 0)

        # Cache reshape op (decode path uses PagedAttention; prefill may update caches)
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        # Default FA op: preserves original behavior (usually TH layout).
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

        # Separate FA op for 2D+ chunk prefill using ragged TND layout and causal mask.
        # Keeping it separate ensures we do not perturb the normal path.
        self._flash_attention_tnd = FlashAttentionScore(
            head_num=head_num,         # Q heads (Nq)
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=0,             # prefill: no look-ahead across chunks
            inner_precise=0,
            input_layout="TND",        # (T, N, D)
            sparse_mode=3,             # causal optimized
        )

        # Decode kernel
        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim
        )

    # ---------- helpers for chunked prefill ----------

    def _is_chunk_prefill(self, q_seq_lens, actual_seq_qlen, actual_seq_kvlen):
        """
        Decide if we are in 2D+ chunk prefill mode.

        Prefer explicit ragged descriptors (actual_seq_qlen/kvlen). If they are not
        present, fall back to a simple multi-entry q_seq_lens check.
        """
        if not self.is_prefill:
            return False

        if (actual_seq_qlen is not None) and (actual_seq_kvlen is not None):
            # If either descriptor has more than one entry, treat as ragged/chunked.
            try:
                if actual_seq_qlen.shape and actual_seq_qlen.shape[0] > 1:
                    return True
                if actual_seq_kvlen.shape and actual_seq_kvlen.shape[0] > 1:
                    return True
            except AttributeError:
                # Scalars or python lists – be conservative and fall through
                pass

        if q_seq_lens is not None:
            try:
                return (q_seq_lens.shape and q_seq_lens.shape[0] > 1)
            except AttributeError:
                return False

        return False

    def _to_tnd(self, x, n_heads, head_dim):
        """Convert packed (T, H*D) -> (T, N, D) for TND FA. If already TND, return as-is."""
        if x is None:
            return None
        shp = x.shape
        if len(shp) == 2 and shp[1] == n_heads * head_dim:
            return ops.reshape(x, (shp[0], n_heads, head_dim))
        return x

    def _from_tnd(self, x):
        """Convert (T, N, D) -> (T, N*D) to match the surrounding packed layout."""
        if x is None:
            return None
        shp = x.shape
        if len(shp) == 3:
            return ops.reshape(x, (shp[0], shp[1] * shp[2]))
        return x

    # ---------- forward ----------

    def construct(self,
                  query,
                  key,
                  value,
                  key_cache,
                  value_cache,
                  slot_mapping,
                  block_tables,
                  batch_valid_length,
                  padding_mask,
                  attn_mask,
                  prefix,
                  q_seq_lens,
                  actual_seq_qlen,
                  actual_seq_kvlen):
        """
        Forward.

        Parameters
        ----------
        query, key, value : Tensor
            Packed TH layout by default: (T, H*D).
        key_cache, value_cache : Tensor
            KV caches for decode (PagedAttention).
        slot_mapping : Tensor
            Indices for cache writes during prefill.
        block_tables : Tensor
            PagedAttention block mapping.
        batch_valid_length : Tensor
            Valid lengths for decode.
        padding_mask, attn_mask, prefix : Tensor or None
            Masks and prefix indicators (passed through to FlashAttentionScore).
        q_seq_lens, actual_seq_qlen, actual_seq_kvlen : Tensor or None
            Ragged/2D prefill descriptors.

        Returns
        -------
        context : Tensor
            Same packed layout as `query`: (T, H*D).
        """
        # Update caches in prefill when MLA not in use (original behavior).
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        if self.is_prefill:
            # 2D+ chunk prefill fast-path using TND layout + causal sparse mode.
            if self._is_chunk_prefill(q_seq_lens, actual_seq_qlen, actual_seq_kvlen):
                q_tnd = self._to_tnd(query, self.head_num, self.hidden_size_per_attention_head)
                kv_heads = self.kv_head_num if self.kv_head_num is not None else self.head_num
                k_tnd = self._to_tnd(key,   kv_heads, self.hidden_size_per_attention_head)
                v_tnd = self._to_tnd(value, kv_heads, self.hidden_size_per_attention_head)

                # real_shift=None, drop_mask=None
                _, _, _, context = self._flash_attention_tnd(
                    q_tnd, k_tnd, v_tnd,
                    None, None,
                    padding_mask, attn_mask, prefix,
                    actual_seq_qlen, actual_seq_kvlen
                )
                context = self._from_tnd(context)
            else:
                # Original prefill path (packed TH)
                # real_shift=None, drop_mask=None
                _, _, _, context = self.flash_attention(
                    query, key, value,
                    None, None,
                    padding_mask, attn_mask, prefix,
                    actual_seq_qlen, actual_seq_kvlen
                )
        else:
            # Decode path (PagedAttention)
            if self.use_multi_latent_attention:
                context = self.paged_attention(
                    query, key_cache, key_cache,
                    block_tables, batch_valid_length,
                    None, None, attn_mask, q_seq_lens
                )
            else:
                context = self.paged_attention(
                    query, key_cache, value_cache,
                    block_tables, batch_valid_length,
                    None, None, attn_mask, q_seq_lens
                )

        return context

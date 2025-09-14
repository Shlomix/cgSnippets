# ------------------- mindformers/modules/flash_attention.py -------------------
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
# -----------------------------------------------------------------------------
"""Flash Attention Layer (TND layout for FA; gather K/V from cache for 2D+ prefill)."""
__all__ = ['FlashAttention']

from mindspore import ops
from mindspore import dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """Flash Attention Layer with 2D+ prefill support via contiguous K/V gather.

    Notes:
      • The Ascend kernel build here does NOT support TH with dim_num=2.
        We therefore run FlashAttentionScore with input_layout="TND" and reshape Q/K/V locally.
      • PagedAttention path stays unchanged and always works because we call ReshapeAndCache first.

    Args:
        head_num (int): Number of attention heads in this partition.
        head_dim (int, optional): Per-head hidden size.
        kv_head_num (int, optional): Number of KV heads in this partition.
        keep_prob (float): Dropout keep prob.
        scale_value (float): Scale for dot-product.
        pre_tokens (int): Sparse window (unused here).
        next_tokens (int): Sparse window (unused here).
        sparse_mode (int): 0 = dense (usual).
        input_layout (str): Ignored; FA is forced to "TND".
        pa_kv_head_num (int, optional): KV heads for PagedAttention op.
        pa_mla_v_dim (int): multi-latent V dim (kept for compatibility).

    Inputs to construct():
        query, key, value:        prefill Q/K/V in TH 2-D [T, H_total] (per partition).
        slot_mapping, block_tables, batch_valid_length, q_seq_lens, actual_seq_qlen, actual_seq_kvlen:
                                  usual paged inputs.
        attn_mask, padding_mask, prefix: passed-through (padding_mask must be None for FA).
        key_cache, value_cache:   paged caches with shape [num_blocks, block_size, kv_heads, head_dim] (or NZ variant).

    Behavior:
        • Always run ReshapeAndCache() first to write current chunk into caches.
        • If `self.is_prefill`: run FA on current chunk (1st prefill) in TND.
        • Else if “2D+ prefill” is detected (q_seq_lens present and max(q_seq_lens)>1) AND caches are present:
            – Gather contiguous K/V up to total_kv_len (from cache + block_tables), reshape to TND, run FA.
        • Otherwise: run PagedAttention (decode and generic fallback).
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
        input_layout="TH",            # ignored; we force TND for FA
        pa_kv_head_num=None,
        pa_mla_v_dim=0,
    ):
        super().__init__()
        self.head_num = int(head_num)
        self.hidden_size_per_attention_head = int(head_dim) if head_dim is not None else None
        self.kv_head_num = int(kv_head_num) if kv_head_num is not None else None
        self.sparse_mode = sparse_mode
        self.is_prefill = True
        self.use_multi_latent_attention = pa_mla_v_dim > 0

        # ---- Ops ----
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.concat0 = ops.Concat(axis=0)
        self.gather = ops.Gather()
        self.reduce_max = ops.ReduceMax(keep_dims=False)
        self.scalar_to_tensor = ops.ScalarToTensor()
        self.logical_and = ops.LogicalAnd()
        self.zeros_like = ops.ZerosLike()
        self.ones_like = ops.OnesLike()

        # Writes the current chunk (K,V) into the paged caches.
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        # Force FA to "TND" to avoid "not support input_layout TH with dim_num 2".
        self.fa_input_layout = "TND"
        self.flash_attention = FlashAttentionScore(
            head_num=self.head_num,            # number of Q heads in this partition
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.fa_input_layout,  # <— TND
            sparse_mode=self.sparse_mode
        )

        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim
        )

    # ---------------- small helpers ----------------

    def _p(self, *a):
        # Tiny gated print; adjust or remove as you like.
        try:
            print(*a)
        except Exception:
            pass

    def _is_second_plus_chunk(self, q_seq_lens):
        """Heuristic: 2D+ prefill if q_seq_lens exists and any element > 1."""
        if q_seq_lens is None:
            return self.scalar_to_tensor(False, mstype.bool_)
        mx = self.reduce_max(q_seq_lens)
        return ops.greater(mx, self.scalar_to_tensor(1, q_seq_lens.dtype))

    def _to_tnd(self, x_2d, n_heads, head_dim):
        """[T, H_total] -> [T, n_heads, head_dim] (TND)."""
        # Let reshape infer T; we validate H_total compat via -1 product.
        return self.reshape(x_2d, (-1, n_heads, head_dim))

    def _harmonize_with_cache_dtype(self, q, k, v, key_cache, value_cache):
        """Ensure q/k/v dtypes match caches when caches exist."""
        if key_cache is not None:
            tgt = key_cache.dtype
            q = self.cast(q, tgt)
            k = self.cast(k, tgt)
            v = self.cast(v, tgt)
        return q, k, v

    def _gather_contiguous_kv(self, kv_cache, block_tables, total_kv_len):
        """
        Gather a contiguous [T_kv, kv_heads, head_dim] from paged cache for a single batch.
        Shapes:
          kv_cache:   [num_blocks, block_size, kv_heads, head_dim]
          block_tbls: [B, max_blocks], 1-based IDs, padded with 0
          total_kv_len: scalar python int (last element of actual_seq_kvlen)
        """
        # Pull static dims from cache shape.
        num_blocks, block_size = int(kv_cache.shape[0]), int(kv_cache.shape[1])
        kv_heads, head_dim = int(kv_cache.shape[2]), int(kv_cache.shape[3])

        # How many blocks do we need to cover total_kv_len?
        needed_blocks = (int(total_kv_len) + block_size - 1) // block_size

        # Slice first row’s first `needed_blocks` entries and convert 1-based -> 0-based.
        blocks_1b = block_tables[0][:needed_blocks]
        blocks_0b = blocks_1b - self.ones_like(blocks_1b)

        # Gather those blocks (axis=0), then flatten to 2-D [T_kv, kv_heads*head_dim].
        gathered = self.gather(kv_cache, blocks_0b, 0)           # [needed_blocks, block, kv_h, d]
        flat = self.reshape(gathered, (-1, kv_heads * head_dim))  # [T_kv, kv_h * d]
        # Finally to TND [T_kv, kv_heads, head_dim]
        flat_tnd = self.reshape(flat, (-1, kv_heads, head_dim))
        return flat_tnd

    # ---------------- main ----------------

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

        # Always push current K/V chunk into the paged cache so decode path is valid.
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # ---------------- First prefill: run FA on the current chunk ----------------
        if self.is_prefill:
            # Expect Q/K/V as 2-D TH [T, H_total]; reshape to TND.
            n_q, n_kv = int(self.head_num), int(self.kv_head_num)
            d = int(self.hidden_size_per_attention_head)

            q_tnd = self._to_tnd(query, n_q, d)
            k_tnd = self._to_tnd(key,   n_kv, d)
            v_tnd = self._to_tnd(value, n_kv, d)

            # Match cache dtype if caches exist (prevents dtype mismatch later).
            q_tnd, k_tnd, v_tnd = self._harmonize_with_cache_dtype(q_tnd, k_tnd, v_tnd, key_cache, value_cache)

            # Kernel currently requires padding_mask=None.
            _, _, _, context = self.flash_attention(
                q_tnd, k_tnd, v_tnd,
                None, None,          # alibi, rel_shift
                None,                # padding_mask (must be None)
                attn_mask,
                prefix,
                actual_seq_qlen,
                actual_seq_kvlen,
            )
            # self._p("[FA] 1st prefill -> context:", context)
            return context

        # ---------------- 2D+ prefill via FA-on-contiguous-KV (decode phase in driver) ----------------
        is_second_plus = self._is_second_plus_chunk(q_seq_lens)
        can_fa = (key_cache is not None) and (value_cache is not None) and (block_tables is not None) \
                 and (actual_seq_kvlen is not None)

        if is_second_plus and can_fa:
            # Use total KV length from last element; convert to python int for gather sizing.
            # (Only B=1 path is used here in PoC.)
            total_kv_len = int(actual_seq_kvlen.asnumpy()[-1])

            # Build contiguous K/V from cache and reshape Q/K/V to TND.
            k_full_tnd = self._gather_contiguous_kv(key_cache,   block_tables, total_kv_len)
            v_full_tnd = self._gather_contiguous_kv(value_cache, block_tables, total_kv_len)

            n_q, d = int(self.head_num), int(self.hidden_size_per_attention_head)
            q_tnd = self._to_tnd(query, n_q, d)

            # Match cache dtype.
            q_tnd, k_full_tnd, v_full_tnd = self._harmonize_with_cache_dtype(q_tnd, k_full_tnd, v_full_tnd,
                                                                              key_cache, value_cache)

            # Run FA on [T_q, n_q, d] × [T_kv, n_kv, d]; padding_mask must be None.
            _, _, _, context = self.flash_attention(
                q_tnd, k_full_tnd, v_full_tnd,
                None, None,          # alibi, rel_shift
                None,                # padding_mask (must be None)
                attn_mask,
                prefix,
                actual_seq_qlen,
                actual_seq_kvlen,
            )
            # self._p("[FA] 2D+ prefill -> context:", context)
            return context

        # ---------------- Default / decode path: PagedAttention on cached KV ----------------
        if self.use_multi_latent_attention:
            context = self.paged_attention(
                query, key_cache, key_cache,
                block_tables, batch_valid_length,
                None, None, attn_mask, q_seq_lens
            )
        else:
            # Keep dtype consistent with caches (common kernel constraint).
            if key_cache is not None:
                query = self.cast(query, key_cache.dtype)
            context = self.paged_attention(
                query, key_cache, value_cache,
                block_tables, batch_valid_length,
                None, None, attn_mask, q_seq_lens
            )
        return context
# ----------------------------------------------------------------------------- end of file

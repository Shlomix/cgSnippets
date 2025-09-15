# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
# ============================================================================
"""Flash Attention Layer"""
__all__ = ['FlashAttention']

from mindspore import ops
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore import dtype as mstype
from mindspore import Tensor


class FlashAttention(Cell):
    """Flash Attention + Paged Attention with 2D+ chunk prefill support.

    Behaviors:
      • First prefill: FlashAttention(q, k, v)   [TH 2-D path as stock]
      • 2D+ prefill:   Gather KV from paged cache using block_tables → FlashAttention(q, Kfull, Vfull)
      • Decode:        PagedAttention(query, key_cache, value_cache)

    Dtype rules:
      • Inputs to FA are harmonized to a stable op dtype (prefer cache dtype if caches exist, else query.dtype).
      • Inputs to PA are explicitly cast so query.dtype == cache.dtype.
      • Returned context is always cast back to query.dtype (matches stock pipeline expectations).
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
        self.is_prefill = True            # set by caller
        self.input_layout = input_layout  # keep "TH" like stock FA
        self.use_multi_latent_attention: bool = pa_mla_v_dim > 0

        # kernels
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

        # utils
        self._print = ops.Print()
        self._zero_i32 = Tensor(0, mstype.int32)
        self._one_i32 = Tensor(1, mstype.int32)

    # ---------------- helpers ----------------

    def _p(self, *xs):
        """lightweight debug print"""
        self._print(*xs)

    def _cast_to(self, x, dt):
        """Cast x to dt only if needed (avoids in-place issues)."""
        return x if x.dtype == dt else ops.cast(x, dt)

    def _is_second_plus_chunk(self, q_seq_lens):
        """
        Detector for 2D+ prefill that does NOT rely on self.is_prefill.
        If any query sub-length > 1, we consider it a 2D+ prefill chunk.
        """
        if q_seq_lens is None:
            return Tensor(False, mstype.bool_)
        return ops.greater(ops.reduce_max(q_seq_lens.astype(mstype.int32)), self._one_i32)

    def _to_i32_scalar(self, x):
        """Ensure we have an int32 scalar Tensor from list/tuple/Tensor."""
        if isinstance(x, Tensor):
            return ops.cast(x, mstype.int32)
        return Tensor(int(x), mstype.int32)

    def _gather_contiguous_kv(self, kv_cache, block_tables, total_kv_len_i32):
        """
        Build a contiguous [T_kv, (H_kv*D)] view from paged cache for a single batch.

        kv_cache shape (non-NZ layout):
            [num_blocks, block_size, kv_heads, head_dim]
        block_tables shape:
            [B=1, max_blocks]  (1-based block indices, padded with 0)
        total_kv_len_i32:
            scalar int32, total tokens currently in KV for this sequence.

        Returns:
            flat [T_kv, kv_heads*head_dim]  (TH 2-D for FA)
        """
        block_size = kv_cache.shape[1]
        # needed_blocks = ceil(total_kv_len / block_size)
        needed_blocks = ops.div(total_kv_len_i32 + (block_size - 1), block_size)  # int

        # Slice first N non-zero entries from block_tables and convert 1-based -> 0-based
        blocks_1b = block_tables[:, :needed_blocks]  # [1, N]
        nz_mask = ops.not_equal(blocks_1b, self._zero_i32)
        # Replace any zeros with 1 to keep gather indices in range; we’ll trim later by exact length.
        blocks_1b = ops.select(nz_mask, blocks_1b, self._one_i32)
        blocks_0b = blocks_1b - self._one_i32  # [1, N], 0-based

        # Gather along block axis (axis 0)
        # kv_cache: [num_blocks, block_size, Hkv, Dh]
        gathered = ops.gather(kv_cache, blocks_0b.reshape((-1,)), 0)  # [N, block_size, Hkv, Dh]
        flat = ops.reshape(gathered, (-1, kv_cache.shape[2] * kv_cache.shape[3]))  # [N*block_size, Hkv*Dh]

        # Trim to exact total_kv_len tokens
        t_kv = int(total_kv_len_i32.asnumpy().item())
        flat = flat[:t_kv]
        return flat  # [T_kv, Hkv*Dh]

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
        """Forward process with first-prefill, 2D+ prefill, and decode paths."""

        # As in stock: write current chunk into caches first
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # Always return the same dtype as incoming query (matches stock Attention pipeline).
        ret_dt = query.dtype

        # Choose a stable op dtype for FA calls (prefer cache dtype if available)
        act_dt = (key_cache.dtype if key_cache is not None else
                  (value_cache.dtype if value_cache is not None else query.dtype))

        # Canonicalize inputs for FA path
        q_act = self._cast_to(query, act_dt)
        k_act = self._cast_to(key,    act_dt)
        v_act = self._cast_to(value,  act_dt)
        am_act = None if attn_mask is None else self._cast_to(attn_mask, act_dt)

        # Decide path
        is_2p = self._is_second_plus_chunk(q_seq_lens)
        first_prefill = ops.logical_and(self.is_prefill, ops.logical_not(is_2p))

        # ---- First prefill → FlashAttention(q, k, v) ----
        if first_prefill:
            # Debug (compact)
            # self._p("[PATH] prefill-1 FA:", q_act.dtype, k_act.dtype, v_act.dtype)
            _, _, _, context = self.flash_attention(
                q_act, k_act, v_act,
                None, None,
                padding_mask, am_act,
                prefix, actual_seq_qlen, actual_seq_kvlen
            )
            context = self._cast_to(context, ret_dt)
            return context

        # ---- 2D+ prefill → gather KV from cache → FlashAttention(q, Kfull, Vfull) ----
        if is_2p:
            # total KV length is cumulative at the last position
            total_kv_len_i32 = self._to_i32_scalar(actual_seq_kvlen[-1])
            k_full = self._gather_contiguous_kv(key_cache,   block_tables, total_kv_len_i32)
            v_full = self._gather_contiguous_kv(value_cache, block_tables, total_kv_len_i32)

            k_full = self._cast_to(k_full, act_dt)
            v_full = self._cast_to(v_full, act_dt)

            # self._p("[PATH] prefill-2+ FA:", q_act.dtype, k_full.dtype, v_full.dtype)
            _, _, _, context = self.flash_attention(
                q_act, k_full, v_full,
                None, None,
                padding_mask, am_act,
                prefix, actual_seq_qlen, actual_seq_kvlen
            )
            context = self._cast_to(context, ret_dt)
            return context

        # ---- Decode → PagedAttention (query must match cache dtype) ----
        cache_dt = key_cache.dtype if key_cache is not None else (value_cache.dtype if value_cache is not None else act_dt)
        query_pa = self._cast_to(q_act, cache_dt)
        am_pa = None if attn_mask is None else self._cast_to(attn_mask, cache_dt)

        if self.use_multi_latent_attention:
            context = self.paged_attention(
                query_pa, key_cache, key_cache,
                block_tables, batch_valid_length, None, None, am_pa, q_seq_lens
            )
        else:
            context = self.paged_attention(
                query_pa, key_cache, value_cache,
                block_tables, batch_valid_length, None, None, am_pa, q_seq_lens
            )
        context = self._cast_to(context, ret_dt)
        return context

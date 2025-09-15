# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
"""Flash Attention Layer"""
__all__ = ['FlashAttention']

from mindspore import ops, Tensor
from mindspore import dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """
    FlashAttention + PagedAttention with a 2D+ prefill fast path:
      - Prefill (first chunk): FA on (q,k,v) directly (TH 2-D)
      - Prefill (2D+): gather K/V from paged cache -> FA
      - Decode: PA on caches, forcing query dtype to match caches
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

        # kernels
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=float(scale_value),
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.input_layout,
            sparse_mode=self.sparse_mode)
        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, float(scale_value), pa_kv_head_num, mla_v_dim=pa_mla_v_dim)

        # utilities
        self._print = ops.Print()
        self._zero_i32 = Tensor(0, mstype.int32)
        self._one_i32 = Tensor(1, mstype.int32)

    # ---------- tiny debug ----------
    def _p(self, *args):
        self._print(*args)

    # ---------- detectors / helpers ----------
    def _is_second_plus_chunk(self, q_seq_lens):
        if q_seq_lens is None:
            return Tensor(False, mstype.bool_)
        return ops.greater(ops.reduce_max(q_seq_lens.astype(mstype.int32)), self._one_i32)

    def _gather_contiguous_kv(self, kv_cache, block_tables, total_kv_len):
        """
        kv_cache: [num_blocks, block_size, kv_heads, head_dim]
        block_tables: [1, max_blocks] (1-based, 0 padded)
        return: flat [T_kv, kv_heads*head_dim] in TH(2-D) form
        """
        block_size = kv_cache.shape[1]
        needed_blocks = ops.div((total_kv_len + (block_size - 1)), block_size)

        blocks_1b = block_tables[:, :needed_blocks]                  # [1, N]
        mask_nz = ops.not_equal(blocks_1b, self._zero_i32)
        blocks_1b = ops.select(mask_nz, blocks_1b, self._one_i32)    # avoid zeros
        blocks_0b = blocks_1b - self._one_i32                         # 0-based

        gathered = ops.gather(kv_cache, blocks_0b.reshape((-1,)), 0)  # [N, BS, Hkv, Dh]
        flat = ops.reshape(gathered, (-1, kv_cache.shape[2] * kv_cache.shape[3]))  # [N*BS, Hkv*Dh]
        flat = flat[:int(total_kv_len)]
        return flat

    def _cast_like(self, x, ref):
        dt = ref if isinstance(ref, mstype.Type) else ref.dtype
        return ops.cast(x, dt) if x.dtype != dt else x

    # ---------- main ----------
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

        # Always write current chunk into caches (stock behavior)
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # ===== PREFILL: first chunk → FA on (q,k,v) =====
        if self.is_prefill and not self._is_second_plus_chunk(q_seq_lens):
            rt = query.dtype  # use model activation dtype (BF16 in your build)
            q = self._cast_like(query, rt)
            k = self._cast_like(key,   rt)
            v = self._cast_like(value, rt)
            am = None if attn_mask is None else self._cast_like(attn_mask, rt)

            self._p("[PATH] prefill-1 (FA) dtypes:", q.dtype, k.dtype, v.dtype)
            _, _, _, context = self.flash_attention(
                q, k, v,
                None, None,
                padding_mask, am,
                prefix, actual_seq_qlen, actual_seq_kvlen
            )
            self._p("[FA] context (1st prefill)", context.shape, context.dtype)
            return context

        # ===== PREFILL: 2D+ chunk → gather K/V then FA =====
        if self.is_prefill:
            total_kv_len = ops.cast(actual_seq_kvlen[-1], mstype.int32)
            k_full = self._gather_contiguous_kv(key_cache,   block_tables, total_kv_len)
            v_full = self._gather_contiguous_kv(value_cache, block_tables, total_kv_len)

            rt = query.dtype  # keep FA in activation dtype
            q = self._cast_like(query, rt)
            k_full = self._cast_like(k_full, rt)
            v_full = self._cast_like(v_full, rt)
            am = None if attn_mask is None else self._cast_like(attn_mask, rt)

            self._p("[PATH] prefill-2+ (gather+FA) dtypes:", q.dtype, k_full.dtype, v_full.dtype)
            _, _, _, context = self.flash_attention(
                q, k_full, v_full,
                None, None,
                padding_mask, am,
                prefix, actual_seq_qlen, actual_seq_kvlen
            )
            self._p("[FA] context (2D+ prefill)", context.shape, context.dtype)
            return context

        # ===== DECODE: PA on cached KV (match cache dtype) =====
        cache_dt = key_cache.dtype if key_cache is not None else (value_cache.dtype if value_cache is not None else query.dtype)
        query_pa = self._cast_like(query, cache_dt)
        am = None if attn_mask is None else self._cast_like(attn_mask, cache_dt)

        if self.use_multi_latent_attention:
            context = self.paged_attention(
                query_pa, key_cache, key_cache,
                block_tables, batch_valid_length, None,
                None, am, q_seq_lens
            )
        else:
            context = self.paged_attention(
                query_pa, key_cache, value_cache,
                block_tables, batch_valid_length, None,
                None, am, q_seq_lens
            )
        self._p("[PA] decode dtypes:", query_pa.dtype, key_cache.dtype if key_cache is not None else "None")
        return context

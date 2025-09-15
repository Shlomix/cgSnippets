# ------------------- mindformers/modules/flash_attention.py -------------------
# Copyright 2025 Huawei Technologies Co.,
# Licensed under the Apache License, Version 2.0
"""Flash Attention Layer (TND layout for FA; contiguous KV gather for 2D+ prefill)."""
__all__ = ['FlashAttention']

from mindspore import ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """
    Prefill:
      • 1st prefill  -> FlashAttention (TND).
      • 2nd+ prefill -> gather contiguous KV from paged cache -> FlashAttention (TND).
    Decode:
      • PagedAttention.

    We DO NOT pass attn_mask/padding_mask to FA (avoid AddExt broadcast issues).
    """

    def __init__(self,
                 head_num,
                 head_dim=None,
                 kv_head_num=None,
                 keep_prob=1.0,
                 scale_value=1.0,
                 pre_tokens=2147483647,
                 next_tokens=2147483647,
                 sparse_mode=0,
                 input_layout="TH",     # ignored; we force TND for FA
                 pa_kv_head_num=None,
                 pa_mla_v_dim=0):
        super().__init__()
        self.head_num = int(head_num)
        self.hidden_size_per_attention_head = int(head_dim) if head_dim is not None else None
        self.kv_head_num = int(kv_head_num) if kv_head_num is not None else None
        self.sparse_mode = sparse_mode
        self.is_prefill = True
        self.use_multi_latent_attention = (pa_mla_v_dim > 0)

        # Ops
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.reduce_max = ops.ReduceMax(keep_dims=False)
        self.scalar_to_tensor = ops.ScalarToTensor()
        self.greater = ops.Greater()
        self.gather = ops.Gather()

        # Cache writer (paged layout)
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        # FlashAttention uses TND to avoid "TH with dim_num 2" errors.
        self.flash_attention = FlashAttentionScore(
            head_num=self.head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout="TND",
            sparse_mode=self.sparse_mode,
        )

        # Decode kernel
        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim
        )

    # ---------------- helpers ----------------
    def _p(self, tag, x=None):
        try:
            if x is None:
                print(tag)
            else:
                print(tag, x)
        except Exception:
            pass

    def _to_tnd(self, x_2d, n_heads, head_dim):
        """[T, n_heads*head_dim] -> [T, n_heads, head_dim]."""
        return self.reshape(x_2d, (-1, n_heads, head_dim))

    def _harmonize_with_cache_dtype(self, q, k, v, key_cache):
        """If cache exists, cast Q/K/V to cache dtype."""
        if key_cache is not None:
            dt = key_cache.dtype
            q = self.cast(q, dt)
            k = self.cast(k, dt)
            v = self.cast(v, dt)
        return q, k, v

    def _is_second_plus_chunk(self, q_seq_lens):
        """2D+ prefill if q_seq_lens exists and any element > 1."""
        if q_seq_lens is None:
            return False
        mx = self.reduce_max(q_seq_lens)
        return bool(self.greater(mx, self.scalar_to_tensor(1, q_seq_lens.dtype)).asnumpy().item())

    def _gather_contiguous_kv(self, kv_cache, block_tables, total_kv_len):
        """
        kv_cache:   [num_blocks, block_size, kv_heads, head_dim]
        block_tbls: [1, max_blocks], 1-based ids, 0 padded
        return:     [T_kv, kv_heads, head_dim]  (TND without the N axis yet)
        """
        num_blocks, block_size, kv_heads, head_dim = map(int, kv_cache.shape)
        need = (int(total_kv_len) + block_size - 1) // block_size
        # 1-based -> 0-based
        blocks_1b = block_tables[0][:need]
        blocks_0b = blocks_1b - ops.ones_like(blocks_1b)
        gathered = self.gather(kv_cache, blocks_0b, 0)  # [need, block, kv_heads, head_dim]
        flat = self.reshape(gathered, (-1, kv_heads, head_dim))
        return flat[:int(total_kv_len)]  # [T_kv, kv_heads, head_dim]

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

        # Always write the current chunk into paged caches so PA works unconditionally.
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # 1) First prefill: use FA (TND) on current chunk.
        if self.is_prefill:
            n_q = self.head_num
            n_kv = self.kv_head_num
            d = self.hidden_size_per_attention_head

            # TH->[T,N,D]
            q_tnd = self._to_tnd(query, n_q, d)
            k_tnd = self._to_tnd(key,   n_kv, d)
            v_tnd = self._to_tnd(value, n_kv, d)

            q_tnd, k_tnd, v_tnd = self._harmonize_with_cache_dtype(q_tnd, k_tnd, v_tnd, key_cache)

            # IMPORTANT: do not pass masks to FA in TND (avoid AddExt broadcast issues)
            _, _, _, ctx = self.flash_attention(q_tnd, k_tnd, v_tnd,
                                                None, None,  # alibi, rel_shift
                                                None,        # padding_mask
                                                None,        # attn_mask
                                                prefix,
                                                actual_seq_qlen,
                                                actual_seq_kvlen)
            # self._p("FA 1st prefill OK; ctx:", ctx.shape)
            return ctx

        # 2) 2D+ prefill (still during prefill stage in driver): gather KV, then FA (TND)
        if self._is_second_plus_chunk(q_seq_lens) and (key_cache is not None) and (value_cache is not None) \
           and (block_tables is not None) and (actual_seq_kvlen is not None):

            total_kv_len = int(actual_seq_kvlen.asnumpy()[-1])
            k_full = self._gather_contiguous_kv(key_cache,   block_tables, total_kv_len)  # [Tk, Nkv, D]
            v_full = self._gather_contiguous_kv(value_cache, block_tables, total_kv_len)  # [Tk, Nkv, D]

            n_q = self.head_num
            d = self.hidden_size_per_attention_head
            q_tnd = self._to_tnd(query, n_q, d)

            q_tnd, k_full, v_full = self._harmonize_with_cache_dtype(q_tnd, k_full, v_full, key_cache)

            # No masks for FA (TND)
            _, _, _, ctx = self.flash_attention(q_tnd, k_full, v_full,
                                                None, None,
                                                None,
                                                None,
                                                prefix,
                                                actual_seq_qlen,
                                                actual_seq_kvlen)
            # self._p("FA 2D+ prefill OK; ctx:", ctx.shape)
            return ctx

        # 3) Default / decode: PagedAttention on cached KV (mask OK here)
        if self.use_multi_latent_attention:
            ctx = self.paged_attention(query, key_cache, key_cache,
                                       block_tables, batch_valid_length,
                                       None, None, attn_mask, q_seq_lens)
        else:
            if key_cache is not None:
                query = self.cast(query, key_cache.dtype)
            ctx = self.paged_attention(query, key_cache, value_cache,
                                       block_tables, batch_valid_length,
                                       None, None, attn_mask, q_seq_lens)
        return ctx
# ---------------- end of file ----------------

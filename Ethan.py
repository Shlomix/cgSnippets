# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
"""Flash Attention Layer"""
__all__ = ['FlashAttention']

from mindspore import ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """FlashAttention (prefill) + PagedAttention (decode) with robust gating for 2D+ chunk prefill."""

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
        self.input_layout = input_layout
        self.use_multi_latent_attention: bool = pa_mla_v_dim > 0

        # The caller (TransformerLayer) flips this per step.
        self.is_prefill = True

        # Debug switch
        self.debug_print = True

        # ops
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.gather = ops.Gather()
        self.reduce_max = ops.ReduceMax(keep_dims=False)
        self.scalar_to_tensor = ops.scalar_to_tensor

        # Cache writer
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        # Kernels
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.input_layout,   # "TH"
            sparse_mode=self.sparse_mode)

        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim)

    # ----------------- debug print -----------------
    def _p(self, tag, x=None):
        if not self.debug_print:
            return
        if x is None:
            print(tag)
            return
        try:
            print(tag, "shape=", getattr(x, "shape", None), "dtype=", getattr(x, "dtype", None))
        except Exception:
            print(tag, x)

    # ----------------- helpers -----------------
    def _cast_like_query(self, x, query):
        return self.cast(x, query.dtype)

    def _kv_to_cache_dtype(self, k, v, key_cache, value_cache):
        if key_cache is not None:
            k = self.cast(k, key_cache.dtype)
        if value_cache is not None:
            v = self.cast(v, value_cache.dtype)
        return k, v

    def _is_second_plus_chunk(self, q_seq_lens):
        """Detector: max(q_seq_lens) > 1  => 2D+ chunk prefill."""
        if q_seq_lens is None:
            return False
        mx = self.reduce_max(q_seq_lens)
        return bool((mx > self.scalar_to_tensor(1, mstype.int32)).asnumpy())

    def _fa_eligible_th(self, q, k, v):
        """Minimal guard for TH FA kernel to avoid Resize failures."""
        # Must be 2-D TH tensors
        if getattr(q, "ndim", -1) != 2 or getattr(k, "ndim", -1) != 2 or getattr(v, "ndim", -1) != 2:
            return False
        # All same dtype
        if not (q.dtype == k.dtype == v.dtype):
            return False
        # Q length multiple-of-16 is safer for current tiling
        q_len = int(q.shape[0])
        if (q_len % 16) != 0:
            return False
        return True

    def _gather_contiguous_kv(self, kv_cache, block_tables, total_kv_len):
        """
        Build contiguous KV up to total_kv_len from paged cache for *single batch*.
        kv_cache: (num_blocks, block_size, kv_heads, head_dim)
        block_tables: (B, max_blocks) 1-based block ids; 0 = pad; B == 1 here.
        Returns TH 2-D: (T_kv, kv_heads*head_dim)
        """
        num_blocks, block_size, kv_heads, head_dim = kv_cache.shape

        # 1) get positive block ids (1-based) from the single batch row
        blocks_1b = block_tables[0]  # (max_blocks,)
        gt0 = ops.Greater()(blocks_1b, self.scalar_to_tensor(0, mstype.int32))
        idxs1 = ops.masked_select(blocks_1b, gt0)  # (needed_blocks,)
        if idxs1.size == 0:
            return self.reshape(ops.zeros((0, kv_heads * head_dim), kv_cache.dtype), (0, kv_heads * head_dim))

        # 2) to 0-based
        idxs0 = idxs1 - self.scalar_to_tensor(1, mstype.int32)

        # 3) gather and flatten
        gathered = self.gather(kv_cache, idxs0, 0)  # (needed_blocks, block_size, kv_heads, head_dim)
        flat3d = self.reshape(gathered, (-1, kv_heads, head_dim))  # (needed_blocks*block_size, kv_heads, head_dim)
        need = min(int(total_kv_len), int(flat3d.shape[0]))
        flat3d = flat3d[:need]
        flat2d = self.reshape(flat3d, (flat3d.shape[0], kv_heads * head_dim))
        return flat2d

    # ----------------- main -----------------
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
        """Forward."""

        # Normalize FA inputs to query dtype for stability.
        key = self._cast_like_query(key, query)
        value = self._cast_like_query(value, query)
        if attn_mask is not None:
            attn_mask = self._cast_like_query(attn_mask, query)

        # ===== PREFILL =====
        if self.is_prefill:
            self._p("[PATH] prefill-1st (FA) candidate q/k/v", query); self._p("k", key); self._p("v", value)

            # Guard: if FA not eligible even for first chunk, fall back to PA (rare, but robust).
            if not self._fa_eligible_th(query, key, value):
                self._p("[GUARD] FA ineligible at 1st prefill -> PagedAttention fallback")
                query_pa = self.cast(query, key_cache.dtype) if key_cache is not None else query
                if self.use_multi_latent_attention:
                    ctx = self.paged_attention(query_pa, key_cache, key_cache,
                                               block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)
                else:
                    ctx = self.paged_attention(query_pa, key_cache, value_cache,
                                               block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)
                return ctx

            # 1st prefill with FA (TH, 2-D)
            _, _, _, context = self.flash_attention(query, key, value,
                                                    None, None,
                                                    padding_mask, attn_mask,
                                                    prefix, actual_seq_qlen,
                                                    actual_seq_kvlen)
            self._p("[FA] context (1st prefill)", context)

            # Write current chunk into caches (cast at boundary so JIT signature stays on query dtype for FA)
            if not self.use_multi_latent_attention and (key_cache is not None and value_cache is not None):
                k_for_cache, v_for_cache = self._kv_to_cache_dtype(key, value, key_cache, value_cache)
                self.reshape_and_cache(k_for_cache, v_for_cache, key_cache, value_cache, slot_mapping)

            return context

        # ===== 2D+ PREFILL (chunk prefill) =====
        is_second_plus = self._is_second_plus_chunk(q_seq_lens)
        if is_second_plus and (key_cache is not None) and (value_cache is not None) and (block_tables is not None) \
           and (actual_seq_kvlen is not None) and (actual_seq_kvlen.size > 0):

            total_kv = int(actual_seq_kvlen[-1])
            k_full = self._gather_contiguous_kv(key_cache,   block_tables, total_kv)
            v_full = self._gather_contiguous_kv(value_cache, block_tables, total_kv)
            k_full = self._cast_like_query(k_full, query)
            v_full = self._cast_like_query(v_full, query)
            self._p("[PATH] prefill-2D+ gathered K/V", k_full); self._p("V", v_full)

            # Guard: only run FA if 2-D & size-friendly; else PA fallback (still prefill step).
            if self._fa_eligible_th(query, k_full, v_full):
                _, _, _, context = self.flash_attention(query, k_full, v_full,
                                                        None, None,
                                                        padding_mask, attn_mask,
                                                        prefix, actual_seq_qlen,
                                                        actual_seq_kvlen)
                self._p("[FA] context (2D+ prefill)", context)
                return context
            else:
                self._p("[GUARD] FA ineligible at 2D+ prefill -> PagedAttention fallback")
                query_pa = self.cast(query, key_cache.dtype)
                if self.use_multi_latent_attention:
                    context = self.paged_attention(query_pa, key_cache, key_cache,
                                                   block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)
                else:
                    context = self.paged_attention(query_pa, key_cache, value_cache,
                                                   block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)
                return context

        # ===== DECODE =====
        # PagedAttention requires dtypes to match the cache; cast only query accordingly.
        query_pa = self.cast(query, key_cache.dtype) if key_cache is not None else query
        self._p("[PATH] decode (PA) query_pa", query_pa)

        if self.use_multi_latent_attention:
            context = self.paged_attention(query_pa, key_cache, key_cache,
                                           block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)
        else:
            context = self.paged_attention(query_pa, key_cache, value_cache,
                                           block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)
        self._p("[PA] decode context", context)
        return context

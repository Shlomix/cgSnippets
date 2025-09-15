# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
"""Flash Attention Layer"""
__all__ = ['FlashAttention']

from mindspore import ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore

class FlashAttention(Cell):
    """Flash Attention + Paged Attention bridge with stable dtypes & simple 2D+ prefill FA.

    Key design:
      - For any FlashAttentionScore call, q/k/v and masks are cast to the SAME dtype as `query`.
      - Cache writes/reads are cast at the boundary to the cache dtype (no JIT signature flips).
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

        # debug switch (env toggle or hardcoded)
        self.debug_print = True

        # ops
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.gather = ops.Gather()
        self.concat = ops.Concat(axis=0)
        self.less_equal = ops.LessEqual()
        self.reduce_max = ops.ReduceMax(keep_dims=False)
        self.logical_and = ops.LogicalAnd()
        self.scalar_to_tensor = ops.scalar_to_tensor
        self.equal = ops.Equal()

        # cache op (K/V -> paged cache)
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        # kernels
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

    # ----------------- small helpers -----------------

    def _p(self, tag, x=None):
        if not self.debug_print:
            return
        if x is None:
            print(tag)
            return
        try:
            print(tag, x.shape, x.dtype)
        except Exception:
            print(tag, x)

    def _is_second_plus_chunk(self, q_seq_lens):
        """Heuristic detector for 2D+ chunk prefill."""
        if q_seq_lens is None:
            return False
        # max(q_seq_lens) > 1
        one = self.scalar_to_tensor(1, mstype.int32)
        mx = self.reduce_max(q_seq_lens)
        return bool((mx > one).asnumpy())

    def _cast_like(self, x, ref):
        tgt = ref.dtype if hasattr(ref, "dtype") else mstype.float16
        return self.cast(x, tgt)

    def _kv_to_cache_dtype(self, k, v, key_cache, value_cache):
        """cast K/V to cache dtype (if caches exist); no-op otherwise."""
        if key_cache is not None:
            k = self.cast(k, key_cache.dtype)
        if value_cache is not None:
            v = self.cast(v, value_cache.dtype)
        return k, v

    def _gather_contiguous_kv(self, kv_cache, block_tables, total_kv_len):
        """
        Build contiguous KV up to total_kv_len from paged cache for *single batch*.

        kv_cache: (num_blocks, block_size, kv_heads, head_dim)
        block_tables: (B, max_blocks) 1-based ids, padded with 0; B==1 here.
        Returns: flat (T_kv, kv_heads*head_dim) in TH 2-D layout.
        """
        # infer needed blocks
        num_blocks = kv_cache.shape[0]
        block_size = kv_cache.shape[1]
        kv_heads   = kv_cache.shape[2]
        head_dim   = kv_cache.shape[3]
        # 0/1-based table -> 0-based positive ids
        blocks_1b = block_tables[0]                         # (max_blocks,)
        # take only positive entries
        is_pos = ops.Greater()(blocks_1b, self.scalar_to_tensor(0, mstype.int32))
        idxs = ops.masked_select(blocks_1b, is_pos)         # (needed_blocks,)
        if idxs.size == 0:
            # empty cache
            return self.reshape(ops.zeros((0, kv_heads*head_dim), kv_cache.dtype), (0, kv_heads*head_dim))
        idxs0 = idxs - self.scalar_to_tensor(1, mstype.int32)

        # gather along axis=0 (blocks)
        gathered = self.gather(kv_cache, idxs0, 0)         # (needed_blocks, block_size, kv_heads, head_dim)
        flat3d   = self.reshape(gathered, (-1, kv_heads, head_dim))  # (needed_blocks*block_size, kv_heads, head_dim)
        total = int(total_kv_len)
        if total < flat3d.shape[0]:
            flat3d = flat3d[:total]
        # TH 2-D (T_kv, kv_heads*head_dim)
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
        """Forward process."""
        # ---- Make FA inputs homogeneous: use query.dtype everywhere for FA path
        q_dtype = query.dtype
        key = self.cast(key, q_dtype)
        value = self.cast(value, q_dtype)
        if attn_mask is not None:
            attn_mask = self.cast(attn_mask, q_dtype)

        # ---------- PREFILL ----------
        if self.is_prefill:
            # 1) first prefill (single chunk): run FA directly in q_dtype
            #    then write K/V into caches (cast at boundary)
            self._p("[FA] prefill=1st :: q/k/v", query); self._p("k", key); self._p("v", value)

            _, _, _, context = self.flash_attention(query, key, value,
                                                    None, None,
                                                    padding_mask, attn_mask,
                                                    prefix, actual_seq_qlen,
                                                    actual_seq_kvlen)
            self._p("[FA] context (1st prefill)", context)

            # Write current chunk into caches for later decode/2D+:
            if not self.use_multi_latent_attention and (key_cache is not None and value_cache is not None):
                k_for_cache, v_for_cache = self._kv_to_cache_dtype(key, value, key_cache, value_cache)
                self.reshape_and_cache(k_for_cache, v_for_cache, key_cache, value_cache, slot_mapping)

            return context

        # ---------- 2D+ PREFILL (chunk prefill) ----------
        # Detect: max(q_seq_lens) > 1
        is_second_plus = self._is_second_plus_chunk(q_seq_lens)
        if is_second_plus and (key_cache is not None) and (value_cache is not None) and (block_tables is not None) \
           and (actual_seq_kvlen is not None) and (actual_seq_kvlen.size > 0):
            # Build contiguous KV from cache, then cast to q_dtype and run FA
            total_kv_len = int(actual_seq_kvlen[-1])
            k_full = self._gather_contiguous_kv(key_cache,   block_tables, total_kv_len)
            v_full = self._gather_contiguous_kv(value_cache, block_tables, total_kv_len)
            k_full = self.cast(k_full, q_dtype)
            v_full = self.cast(v_full, q_dtype)
            self._p("[FA] 2D+ gathered K", k_full); self._p("[FA] 2D+ gathered V", v_full)

            _, _, _, context = self.flash_attention(query, k_full, v_full,
                                                    None, None,
                                                    padding_mask, attn_mask,
                                                    prefix, actual_seq_qlen,
                                                    actual_seq_kvlen)
            self._p("[FA] context (2D+ prefill)", context)
            return context

        # ---------- DECODE (paged attention) ----------
        # Paged attention expects query to match cache dtype; cast only query.
        if key_cache is not None:
            query_pa = self.cast(query, key_cache.dtype)
        else:
            query_pa = query  # fallback

        if self.use_multi_latent_attention:
            context = self.paged_attention(query_pa, key_cache, key_cache,
                                           block_tables, batch_valid_length, None,
                                           None, attn_mask, q_seq_lens)
        else:
            context = self.paged_attention(query_pa, key_cache, value_cache,
                                           block_tables, batch_valid_length, None,
                                           None, attn_mask, q_seq_lens)

        self._p("[PA] decode context", context)
        return context

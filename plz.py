# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
"""Flash Attention Layer"""

__all__ = ['FlashAttention']

from mindspore import ops
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """
    Prefill:
      - 1st prefill  -> FlashAttention (TH 2-D)
      - 2nd+ prefill -> FlashAttention if we can gather contiguous KV from cache,
                        else fall back to PagedAttention
    Decode:
      - PagedAttention (paged KV cache)
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
                 input_layout="TH",   # TH 2-D (T, H)
                 pa_kv_head_num=None,
                 pa_mla_v_dim=0):
        super().__init__()
        self.head_num = head_num
        self.hidden_size_per_attention_head = head_dim
        self.kv_head_num = kv_head_num
        self.sparse_mode = sparse_mode

        # These flags are set by Attention before each call.
        self.is_prefill: bool = True
        self.debug_layer: int = -1  # -1 = no prints

        # Optional: when True and conditions met, we try FA on 2nd+ prefill chunks
        self.enable_chunk2p_fa: bool = True

        self.use_multi_latent_attention: bool = pa_mla_v_dim > 0

        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()
        self.paged_attention = ops.auto_generate.PagedAttention(
            head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim)

        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=input_layout,   # keep TH
            sparse_mode=self.sparse_mode)

        self._cast = ops.Cast()
        self._concat = ops.Concat(axis=0)
        self._reshape = ops.Reshape()
        self._gather = ops.Gather()
        self._zeros_like = ops.ZerosLike()

    # ---------- tiny scoped print helper ----------
    def _p(self, tag, obj=None):
        if self.debug_layer < 0:
            return
        print(f"[FLASH L{self.debug_layer}] {tag}")
        if obj is None:
            return
        try:
            if hasattr(obj, "shape"):
                print(f"Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype})")
            else:
                print(obj)
        except Exception:
            print(obj)

    # ---------- small utilities ----------
    def _dtype_harmonize(self, q, k, v, key_cache, value_cache):
        """If caches exist, cast Q/K/V to cache dtype to satisfy op type checks."""
        if key_cache is not None and value_cache is not None:
            tgt = key_cache.dtype
            if q.dtype != tgt: q = self._cast(q, tgt)
            if k.dtype != tgt: k = self._cast(k, tgt)
            if v.dtype != tgt: v = self._cast(v, tgt)
        return q, k, v

    def _is_second_plus_chunk(self, q_seq_lens, batch_valid_length, actual_seq_kvlen):
        """
        Heuristic used in your traces:
          - packed prefill: sum(q_seq_lens) > batch_size
          - and total KV just grew beyond batch_valid_length
        """
        try:
            if q_seq_lens is None or batch_valid_length is None or actual_seq_kvlen is None:
                return False
            bsz = int(q_seq_lens.shape[0])
            has_chunk = int(q_seq_lens.sum()) > bsz
            grew = int(actual_seq_kvlen[-1]) > int(batch_valid_length[-1])
            return bool(has_chunk and grew)
        except Exception:
            return False

    def _gather_contiguous_kv(self, kv_cache, block_tables, total_kv_len):
        """
        Build a contiguous [T_kv, H_kv] view from paged cache for *single batch*.

        kv_cache: (num_blocks, block_size, kv_heads, head_dim)
        block_tables: (B, max_blocks) with 1-based block ids, padded with 0
        total_kv_len: int, e.g. actual_seq_kvlen[-1]
        """
        # assumptions that match your setup
        assert len(kv_cache.shape) == 4, "kv_cache must be (num_blocks, block_size, kv_heads, head_dim)"
        assert len(block_tables.shape) == 2 and block_tables.shape[0] == 1, "only B=1 supported in this PoC"

        num_blocks, block_size, kv_heads, head_dim = kv_cache.shape
        needed_blocks = (int(total_kv_len) + block_size - 1) // block_size

        # slice the first 'needed_blocks' non-zero entries; convert to 0-based indices
        blocks_1b = block_tables[0, :needed_blocks]          # (needed_blocks,)
        one = ops.ones_like(blocks_1b)
        blocks_0b = blocks_1b - one                          # (needed_blocks,)
        # Gather blocks along axis=0
        gathered = self._gather(kv_cache, blocks_0b, 0)      # (needed_blocks, block_size, kv_heads, head_dim)
        flat = self._reshape(gathered, (needed_blocks * block_size, kv_heads, head_dim))
        flat = flat[:int(total_kv_len)]                      # (T_kv, kv_heads, head_dim)
        flat2d = self._reshape(flat, (int(total_kv_len), kv_heads * head_dim))  # (T_kv, H_kv)
        return flat2d

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
        """
        Inputs follow the parallel_core Attention caller.
        Prefill: Q/K/V are 2-D (T, H) in TH layout per partition.
        """
        self._p(f"is_prefill={self.is_prefill}")
        self._p("q", query); self._p("k", key); self._p("v", value)
        self._p("key_cache", key_cache); self._p("value_cache", value_cache)
        self._p("block_tables", block_tables)
        self._p("q_seq_lens", q_seq_lens)
        self._p("actual_seq_qlen", actual_seq_qlen)
        self._p("actual_seq_kvlen", actual_seq_kvlen)
        self._p("batch_valid_length", batch_valid_length)

        # If caches exist, first write the current chunk into them.
        if (not self.use_multi_latent_attention) and (key_cache is not None) and (value_cache is not None) and (slot_mapping is not None):
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # Keep dtypes consistent with caches when present.
        query, key, value = self._dtype_harmonize(query, key, value, key_cache, value_cache)

        if self.is_prefill:
            # --- First prefill (no cache yet) OR forced 2D+ prefill with gather ---
            second_plus = self._is_second_plus_chunk(q_seq_lens, batch_valid_length, actual_seq_kvlen)
            can_gather = (key_cache is not None) and (value_cache is not None) and (block_tables is not None) and (actual_seq_kvlen is not None)
            use_fa_gather = second_plus and can_gather and self.enable_chunk2p_fa

            if use_fa_gather:
                # Build contiguous K/V up to total kv len and run FA with 2-D TH tensors.
                total_kv_len = int(actual_seq_kvlen[-1])
                k_full = self._gather_contiguous_kv(key_cache, block_tables, total_kv_len)
                v_full = self._gather_contiguous_kv(value_cache, block_tables, total_kv_len)
                # Q is already the current chunk in TH 2-D (T_q, H_q)
                self._p("FA 2D+prefill q", query)
                self._p("FA 2D+prefill k_full", k_full)
                self._p("FA 2D+prefill v_full", v_full)

                # FlashAttentionScore(q, k, v, alibi, rel_shift, padding_mask, attn_mask, prefix, actual_q, actual_kv)
                _, _, _, context = self.flash_attention(query, k_full, v_full,
                                                        None, None,
                                                        None, None,
                                                        None,
                                                        actual_seq_qlen, actual_seq_kvlen)
                self._p("FA context (2D+prefill)", context)
                return context

            else:
                # 1st prefill (no cache) → FA directly on (q, k, v) in TH
                self._p("FA 1st prefill (TH)")
                _, _, _, context = self.flash_attention(query, key, value,
                                                        None, None,
                                                        None, None,
                                                        None,
                                                        actual_seq_qlen, actual_seq_kvlen)
                self._p("FA context (1st prefill)", context)
                return context

        # --- Decode path: PagedAttention on cached KV ---
        # (Query stays 2-D TH (T=1, H), PA accepts cache tensors.)
        context = self.paged_attention(query,
                                       key_cache if key_cache is not None else key,
                                       value_cache if value_cache is not None else value,
                                       block_tables, batch_valid_length,
                                       None, None,
                                       attn_mask, q_seq_lens)
        self._p("PA context (decode)", context)
        return context

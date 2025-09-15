# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0

"""Flash Attention Layer"""
__all__ = ['FlashAttention']

from mindspore import ops, Tensor
from mindspore import dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """Flash Attention Layer.

    Inputs / behaviors are identical to the stock version, but we:
      * detect 2D+ prefill and build contiguous KV, then run FA
      * fix dtype drift across calls by unifying FA/PA input dtype
      * add lightweight debug prints

    Supported Platforms: Ascend
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

        # runtime ops
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=float(scale_value),  # avoid implicit float64
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.input_layout,
            sparse_mode=self.sparse_mode)
        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, float(scale_value), pa_kv_head_num, mla_v_dim=pa_mla_v_dim)

        # small helpers
        self._print = ops.Print()
        self._zero_i32 = Tensor(0, mstype.int32)
        self._one_i32 = Tensor(1, mstype.int32)

    # ---------- debug ----------
    def _p(self, tag, x=None):
        self._print(tag)
        if x is not None:
            # print only small headers to keep logs readable
            if isinstance(x, Tensor):
                self._print("Tensor(shape=", x.shape, ", dtype=", x.dtype, ")")
            else:
                self._print(x)

    # ---------- detectors / helpers ----------
    def _is_second_plus_chunk(self, q_seq_lens):
        """Heuristic: any q-len > 1 indicates 2D+ prefill."""
        if q_seq_lens is None:
            return Tensor(False, mstype.bool_)
        max_q = ops.reduce_max(q_seq_lens.astype(mstype.int32))
        return ops.greater(max_q, self._one_i32)

    def _gather_contiguous_kv(self, kv_cache, block_tables, total_kv_len):
        """
        Build a contiguous [T_kv, H_kv*D] view from paged cache for single batch.
        kv_cache: [num_blocks, block_size, kv_heads, head_dim]
        block_tables: [B=1, max_blocks] with 1-based indices; padded with 0
        total_kv_len: scalar int32 (<= max_blocks * block_size)
        """
        # shape checks that match MF layout
        # num_blocks, block_size, kv_heads, head_dim = kv_cache.shape
        # blocks we need = ceil(total_kv_len / block_size)
        block_size = kv_cache.shape[1]
        needed_blocks = ops.div((total_kv_len + (block_size - 1)), block_size)
        # pick first `needed_blocks` non-zero entries; convert 1-based -> 0-based
        blocks_1b = block_tables[:, :needed_blocks]                 # [1, N]
        mask_nz = ops.not_equal(blocks_1b, self._zero_i32)
        blocks_1b = ops.select(mask_nz, blocks_1b, self._one_i32)   # keep shape, avoid zeros
        blocks_0b = blocks_1b - self._one_i32                        # [1, N]

        # gather along block axis (0)
        # kv_cache: [num_blocks, block_size, kv_heads, head_dim]
        gathered = ops.gather(kv_cache, blocks_0b.reshape((-1,)), 0)  # [N, BS, Hkv, Dh]
        flat = ops.reshape(gathered, (-1, kv_cache.shape[2] * kv_cache.shape[3]))  # [N*BS, Hkv*Dh]
        # cut to exact total_kv_len
        flat = flat[:int(total_kv_len)]
        return flat  # [T_kv, Hkv*Dh] in TH(2D) form

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
        """Forward – supports prefill-1 (FA), prefill-2+ (gather+FA), decode (PA)."""

        # Always write the current chunk to cache when caches exist
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # Pick a single runtime dtype and stick with it across all calls.
        # We force float16 so the very first call matches later decode calls (whose caches are fp16).
        runtime_dtype = mstype.float16

        # =========================
        # Prefill-1: run FA directly
        # =========================
        if self.is_prefill and not self._is_second_plus_chunk(q_seq_lens):
            q = self._cast_like(query, runtime_dtype)
            k = self._cast_like(key,   runtime_dtype)
            v = self._cast_like(value, runtime_dtype)
            am = None if attn_mask is None else self._cast_like(attn_mask, runtime_dtype)

            self._p("[PATH] prefill-1 (FA) dtypes")
            self._p("q", q); self._p("k", k); self._p("v", v)

            _, _, _, context = self.flash_attention(
                q, k, v,
                None, None,
                padding_mask, am,
                prefix, actual_seq_qlen, actual_seq_kvlen
            )
            self._p("[FA] context (1st prefill)", context)
            return context

        # =========================
        # Prefill-2+: gather K/V then FA
        # =========================
        if self.is_prefill:
            # totals are cumulative lengths; use last element
            total_kv_len = ops.cast(actual_seq_kvlen[-1], mstype.int32)

            k_full = self._gather_contiguous_kv(key_cache,   block_tables, total_kv_len)
            v_full = self._gather_contiguous_kv(value_cache, block_tables, total_kv_len)

            q = self._cast_like(query, runtime_dtype)
            k_full = self._cast_like(k_full, runtime_dtype)
            v_full = self._cast_like(v_full, runtime_dtype)
            am = None if attn_mask is None else self._cast_like(attn_mask, runtime_dtype)

            self._p("[PATH] prefill-2+ (gather+FA) dtypes")
            self._p("q", q); self._p("k_full", k_full); self._p("v_full", v_full)

            _, _, _, context = self.flash_attention(
                q, k_full, v_full,
                None, None,
                padding_mask, am,
                prefix, actual_seq_qlen, actual_seq_kvlen
            )
            self._p("[FA] context (2D+ prefill)", context)
            return context

        # =========================
        # Decode: PA on cached KV
        # =========================
        # PagedAttention requires query and caches to match dtype.
        query_pa = self._cast_like(query, runtime_dtype)
        am = None if attn_mask is None else self._cast_like(attn_mask, runtime_dtype)

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
        self._p("[PA] decode context", context)
        return context

# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
"""Flash Attention Layer"""
__all__ = ['FlashAttention']

from mindspore import ops, Tensor
from mindspore import dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """FA for prefill, FA+gather for 2D+ prefill, PA for decode.

    Routing:
      • First prefill:           FlashAttentionScore(q,k,v) in 2-D TH
      • 2D+ prefill (chunking):  gather K/V from paged cache → FlashAttentionScore(q, Kfull, Vfull)
      • Decode:                  PagedAttention(query, key_cache, value_cache)
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
        self.is_prefill = True                         # set by caller at runtime
        self.input_layout = input_layout               # we keep TH for FA
        self.use_multi_latent_attention = pa_mla_v_dim > 0

        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=float(scale_value),
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.input_layout,            # TH
            sparse_mode=self.sparse_mode)

        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, float(scale_value), pa_kv_head_num, mla_v_dim=pa_mla_v_dim)

        self._print = ops.Print()
        self._zero_i32 = Tensor(0, mstype.int32)
        self._one_i32  = Tensor(1, mstype.int32)

    # ---- tiny helpers / debug ----
    def _p(self, *xs): self._print(*xs)

    def _cast_to(self, x, dt):
        return ops.cast(x, dt) if x.dtype != dt else x

    def _is_second_plus_chunk(self, q_seq_lens):
        # purely by q_seq_lens; works even when self.is_prefill == False
        if q_seq_lens is None:
            return Tensor(False, mstype.bool_)
        return ops.greater(ops.reduce_max(q_seq_lens.astype(mstype.int32)), self._one_i32)

    def _gather_contiguous_kv(self, kv_cache, block_tables, total_kv_len):
        """
        kv_cache:    [num_blocks, block_size, kv_heads, head_dim]
        block_tables:[B(=1), max_blocks] (1-based, 0-padded)
        returns flat TH 2-D: [T_kv, kv_heads*head_dim]
        """
        block_size   = kv_cache.shape[1]
        needed_blocks = ops.div((total_kv_len + (block_size - 1)), block_size)

        blocks_1b = block_tables[:, :needed_blocks]                      # [1, N]
        nz_mask  = ops.not_equal(blocks_1b, self._zero_i32)
        blocks_1b = ops.select(nz_mask, blocks_1b, self._one_i32)        # avoid zeros
        blocks_0b = blocks_1b - self._one_i32                             # 0-based

        gathered = ops.gather(kv_cache, blocks_0b.reshape((-1,)), 0)     # [N, BS, H, Dh]
        flat = ops.reshape(gathered, (-1, kv_cache.shape[2] * kv_cache.shape[3]))  # [N*BS, H*Dh]
        flat = flat[:int(total_kv_len)]
        return flat

    # ---- main ----
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

        # stock behavior: write current chunk into caches first
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # --------- DTYPE CANONICALIZATION (fixes “sticky dtype” across iterations) ----------
        # Prefer cache dtype as the activation dtype for FA; it is stable across steps.
        act_dt = (key_cache.dtype if key_cache is not None else
                  (value_cache.dtype if value_cache is not None else query.dtype))

        # Canonical FA inputs live in act_dt (usually BF16 in your builds)
        q_act = self._cast_to(query, act_dt)
        k_act = self._cast_to(key,    act_dt)
        v_act = self._cast_to(value,  act_dt)
        am_act = None if attn_mask is None else self._cast_to(attn_mask, act_dt)

        # PA requires all three tensors to share the cache dtype
        cache_dt = act_dt if key_cache is None else key_cache.dtype

        # --------- ROUTING ----------
        is_2p          = self._is_second_plus_chunk(q_seq_lens)
        first_prefill  = ops.logical_and(self.is_prefill, ops.logical_not(is_2p))

        # ===== Prefill: first chunk → FA(q,k,v) =====
        if first_prefill:
            self._p("[PATH] prefill-1  (FA)  dtypes:", q_act.dtype, k_act.dtype, v_act.dtype)
            _, _, _, context = self.flash_attention(
                q_act, k_act, v_act,
                None, None,
                padding_mask, am_act,
                prefix, actual_seq_qlen, actual_seq_kvlen
            )
            self._p("[FA] context (1st prefill):", context.shape, context.dtype)
            return context

        # ===== Prefill: 2D+ chunk → gather K/V then FA =====
        if is_2p:
            total_kv_len = ops.cast(actual_seq_kvlen[-1], mstype.int32)
            k_full = self._gather_contiguous_kv(key_cache,   block_tables, total_kv_len)
            v_full = self._gather_contiguous_kv(value_cache, block_tables, total_kv_len)

            k_full = self._cast_to(k_full, act_dt)
            v_full = self._cast_to(v_full, act_dt)

            self._p("[PATH] prefill-2+ (gFA) dtypes:", q_act.dtype, k_full.dtype, v_full.dtype)
            _, _, _, context = self.flash_attention(
                q_act, k_full, v_full,
                None, None,
                padding_mask, am_act,
                prefix, actual_seq_qlen, actual_seq_kvlen
            )
            self._p("[FA] context (2D+ prefill):", context.shape, context.dtype)
            return context

        # ===== Decode: PA on cached KV (query cast → cache dtype) =====
        query_pa = self._cast_to(q_act, cache_dt)
        am_pa = None if attn_mask is None else self._cast_to(attn_mask, cache_dt)

        if self.use_multi_latent_attention:
            context = self.paged_attention(
                query_pa, key_cache, key_cache,
                block_tables, batch_valid_length,
                None, None, am_pa, q_seq_lens
            )
        else:
            context = self.paged_attention(
                query_pa, key_cache, value_cache,
                block_tables, batch_valid_length,
                None, None, am_pa, q_seq_lens
            )
        self._p("[PATH] decode     (PA)  dtypes:", query_pa.dtype,
                key_cache.dtype if key_cache is not None else "None")
        return context

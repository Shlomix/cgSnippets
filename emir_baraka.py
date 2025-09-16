# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Flash Attention Layer with 2D+ Chunk Prefill (TND) using KV cache."""
__all__ = ["FlashAttention"]

import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """
    Flash Attention layer used in:
      • non-chunk prefill: FlashAttention (original TH layout path)
      • 2D(+)-chunk prefill: FlashAttention (TND layout, K/V from cache)
      • decode: PagedAttention

    TND rules (MindSpore flash_attention_score):
      - input_layout="TND"
      - pass actual_seq_qlen and actual_seq_kvlen (cumulative; last == T)
      - causal sparse_mode (2 or 3); next_tokens=0; prefix=None
      - attn_mask must be (2048, 2048) lower-tri with 1=discard, 0=keep
    """

    # NOTE: Signature preserved exactly (no arg reordering / defaults added).
    def __init__(self,
                 head_num,
                 kv_head_num,
                 hidden_size_per_attention_head,
                 keep_prob,
                 scale_value,
                 pre_tokens,
                 next_tokens,
                 input_layout,
                 sparse_mode,
                 pa_kv_head_num=None,
                 pa_mla_v_dim=0):
        super().__init__()
        # ----- config -----
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.sparse_mode = sparse_mode
        self.is_prefill = True              # may be flipped by caller; logic below doesn't rely on it
        self.input_layout = input_layout
        self.use_multi_latent_attention = (pa_mla_v_dim > 0)

        # ----- ops -----
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        # Original FA primitive (prefill, TH path)
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
        self.flash_attention.add_prim_attr("mf_role", "fa_prefill_TH")

        # TND FA primitive for 2D(+)-chunk prefill (causal; next_tokens=0)
        self._flash_attention_tnd = FlashAttentionScore(
            head_num=head_num,        # N_q
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=2147483647,    # satisfy pre_tokens >= max_q_len
            next_tokens=0,
            inner_precise=0,
            input_layout="TND",       # (T, N, D)
            sparse_mode=3,            # causal optimized
        )
        self._flash_attention_tnd.add_prim_attr("mf_role", "fa_chunk_TND")

        # Decode kernel unchanged
        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim
        )

        # Handy kernels
        self._reshape = ops.Reshape()
        self._gather = ops.Gather()           # (params, indices, axis)
        self._gather_d = ops.GatherD()        # gather_d(x, dim, index)
        self._masked_select = ops.MaskedSelect()
        self._tile = ops.Tile()
        self._expand_dims = ops.ExpandDims()
        self._reduce_max = ops.ReduceMax()
        self._range = ops.Range()
        self._cast = ops.Cast()
        self._greater = ops.Greater()
        self._concat0 = ops.Concat(axis=0)

        # Canonical 2048x2048 lower-tri discard mask (uint8), built lazily
        self._tnd_mask_2048 = None

    # ------------------ helpers ------------------

    def _has_ragged_descriptors(self, q_seq_lens, actual_seq_qlen, actual_seq_kvlen):
        """
        Treat as chunked/ragged iff both *_actual_seq_* are provided (TND requires them).
        Some pipelines set is_prefill=False during chunk prefill, so we do NOT rely on it.
        """
        if (actual_seq_qlen is None) or (actual_seq_kvlen is None):
            return False
        # Require tensors (rank >= 1)
        if len(actual_seq_qlen.shape) == 0 or len(actual_seq_kvlen.shape) == 0:
            return False
        return True

    def _to_tnd_from_th(self, x, n_heads, head_dim):
        """(T, H*D) -> (T, N, D) for TND FA."""
        if x is None:
            return None
        t, hd = x.shape
        if hd == n_heads * head_dim:
            return self._reshape(x, (t, n_heads, head_dim))
        return x

    def _from_tnd_to_th(self, x):
        """(T, N, D) -> (T, N*D) to match packed layout outside FA."""
        if x is None:
            return None
        if len(x.shape) == 3:
            t, n, d = x.shape
            return self._reshape(x, (t, n * d))
        return x

    def _diff_lengths(self, cum_lengths):
        """
        cumulative [l1, l1+l2, ...] -> per-seq [l1, l2, ...]
        cum_lengths: Tensor[int32 or int64] shape [B]
        """
        B = cum_lengths.shape[0]
        zero = self._cast(self._range(0, 1, 1), cum_lengths.dtype)  # (1,)
        prev = self._concat0((zero, cum_lengths[:B-1]))
        return cum_lengths - prev  # [B]

    def _kv_from_cache_tnd(self, cache, block_tables, actual_seq_kvlen):
        """
        Build (T2, N_kv, D) from block-wise cache using block_tables and ragged kv lengths.
        cache:        (num_blocks, block_size, N_kv, D)
        block_tables: (B, max_blocks_per_seq)
        actual_seq_kvlen: cumulative kv lengths [B], last == T2
        """
        nb, bs, n_kv, d = cache.shape
        # Flatten blocks to token-major view: (nb*bs, N_kv, D)
        flat = self._reshape(cache, (nb * bs, n_kv, d))

        # lengths per sequence [B]
        kv_cum = actual_seq_kvlen
        kv_lens = self._diff_lengths(kv_cum)                  # [B]
        max_len = self._reduce_max(kv_lens)                   # scalar
        max_len_i32 = self._cast(max_len, mstype.int32)

        # positions [0..max_len-1], shape (max_len,)
        pos = self._range(self._cast(0, mstype.int32), max_len_i32, self._cast(1, mstype.int32))
        # tile to (B, max_len)
        B = kv_lens.shape[0]
        pos = self._tile(self._expand_dims(pos, 0), (B, 1))   # [B, max_len]

        # mask valid positions per sequence
        kv_lens_i32 = self._cast(kv_lens, mstype.int32)
        kv_lens_2d = self._expand_dims(kv_lens_i32, 1)        # [B,1]
        valid_mask = self._greater(kv_lens_2d, pos)           # True where pos < len

        # compute global token indices in flattened cache for each (b, pos)
        blk_idx = pos // bs                                   # [B, max_len]
        table_i32 = self._cast(block_tables, mstype.int32)    # [B, max_blocks]
        blk_ids = self._gather_d(table_i32, 1, blk_idx)       # [B, max_len]
        offsets = pos - (blk_idx * bs)                        # [B, max_len]
        global_idx = blk_ids * bs + offsets                   # [B, max_len], int32

        # Select only valid positions (ragged -> packed 1D)
        valid_idx_flat = self._masked_select(global_idx, valid_mask)  # [T2]
        # Gather (T2, N_kv, D)
        kv_tnd = self._gather(flat, valid_idx_flat, 0)
        return kv_tnd  # (T2, N_kv, D)

    def _ensure_tnd_causal_mask(self, attn_mask, size=2048):
        """
        TND requires a lower-triangular mask of shape (2048, 2048),
        dtype bool/uint8, with 1=discard and 0=keep. If incoming mask doesn't
        match, build the canonical one.
        """
        if self._tnd_mask_2048 is None:
            size_i32 = self._cast(size, mstype.int32)
            idx = self._range(self._cast(0, mstype.int32), size_i32, self._cast(1, mstype.int32))  # (size,)
            row = self._tile(self._expand_dims(idx, 1), (size_i32, 1))  # (size, size)
            col = self._tile(self._expand_dims(idx, 0), (1, size_i32))  # (size, size)
            upper = self._greater(col, row)  # True on strictly upper triangle (j > i)
            self._tnd_mask_2048 = self._cast(upper, mstype.uint8)  # 1=discard, 0=keep

        # If supplied mask already (2048,2048) and uint8/bool, use it.
        if attn_mask is not None:
            if len(attn_mask.shape) == 2 and attn_mask.shape[0] == 2048 and attn_mask.shape[1] == 2048:
                if attn_mask.dtype == mstype.uint8 or attn_mask.dtype == mstype.bool_:
                    return attn_mask
        return self._tnd_mask_2048

    def _fa_call(self, fa_prim, q, k, v, padding_mask, attn_mask, prefix, actual_seq_qlen, actual_seq_kvlen):
        """
        Call FlashAttentionScore and return the attention context tensor, regardless
        of whether the primitive returns a Tensor or a tuple.
        """
        out = fa_prim(q, k, v, None, None, padding_mask, attn_mask, prefix, actual_seq_qlen, actual_seq_kvlen)
        if isinstance

# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Flash Attention Layer (TND-only for prefill & chunk prefill)"""
__all__ = ['FlashAttention']

import os
import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """Flash Attention wrapper with TND FA for both prefill and 2D(+)-chunk prefill, plus PagedAttention decode.

    Public API (ctor + construct signature) matches upstream.
    Casting of activations (e.g., to FP16) is handled by the caller.
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
            input_layout="TH",   # accepted but ignored; we run TND internally
            pa_kv_head_num=None,
            pa_mla_v_dim=0
    ):
        super().__init__()
        self.head_num = head_num
        self.hidden_size_per_attention_head = head_dim
        self.kv_head_num = kv_head_num
        self.sparse_mode = sparse_mode
        self.is_prefill = True
        self.input_layout = "TND"   # force TND internally
        self.use_multi_latent_attention: bool = pa_mla_v_dim > 0

        # ---- debug (env-gated) ----
        dbg_env = os.getenv("MF_FA_DEBUG", "").lower()
        self._debug = dbg_env not in ("", "0", "false", "off", "no")
        self._print = ops.Print()

        # --- Core ops ---
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        # TND FA for **prefill** (use caller's next_tokens/sparse_mode/scale_value)
        self.flash_attention_tnd_prefill = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout="TND",
            sparse_mode=self.sparse_mode)
        self.flash_attention_tnd_prefill.add_prim_attr("mf_role", "fa_TND_prefill")

        # TND FA for **2D(+)-chunk prefill** (strict causal; no right lookahead)
        self.flash_attention_tnd_chunk = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=0,          # strict causal for chunked prefill
            inner_precise=0,
            input_layout="TND",
            sparse_mode=3)          # causal optimized
        self.flash_attention_tnd_chunk.add_prim_attr("mf_role", "fa_TND_chunk")

        # Decode kernel (PagedAttention)
        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim)

        # --- helpers (no Tile; minimal casts only for small synthetic ints / cache write alignment) ---
        self._reshape = ops.Reshape()
        self._expand_dims = ops.ExpandDims()
        self._range = ops.Range()
        self._reduce_max = ops.ReduceMax()
        self._greater = ops.Greater()
        self._gather = ops.Gather()
        self._gather_d = ops.GatherD()
        self._masked_select = ops.MaskedSelect()
        self._concat0 = ops.Concat(axis=0)
        self._zeros = ops.Zeros()
        self._ones = ops.Ones()
        self._cast = ops.Cast()
        self._triu_strict = ops.Triu(diagonal=1)  # strictly upper triangle (j > i)

        # Canonical (2048, 2048) causal discard mask for TND
        self._tnd_mask_2048 = None

    # -------------------- debug helper --------------------
    def _dbg(self, msg):
        if self._debug:
            self._print("[MF_FA] " + msg)

    # -------------------- shape/layout helpers --------------------
    def _has_tnd_ragged(self, actual_seq_qlen, actual_seq_kvlen):
        if actual_seq_qlen is None or actual_seq_kvlen is None:
            return False
        if len(actual_seq_qlen.shape) == 0 or len(actual_seq_kvlen.shape) == 0:
            return False
        return True

    def _infer_head_dim(self, query):
        if self.hidden_size_per_attention_head is not None:
            return self.hidden_size_per_attention_head
        return query.shape[1] // self.head_num  # TH-packed incoming: (T, H*D)

    def _to_tnd_from_th(self, x, n_heads, head_dim):
        # (T, H*D) -> (T, N, D)
        if x is None:
            return None
        t, hd = x.shape
        if hd == n_heads * head_dim:
            return self._reshape(x, (t, n_heads, head_dim))
        return x

    def _from_tnd_to_th(self, x):
        # (T, N, D) -> (T, N*D)
        if x is None:
            return None
        if len(x.shape) == 3:
            t, n, d = x.shape
            return self._reshape(x, (t, n * d))
        return x

    def _diff_lengths(self, cum_lengths):
        # cumulative [l1, l1+l2, ...] -> per-seq [l1, l2, ...]; cum_lengths: [B]
        B = cum_lengths.shape[0]
        zero = self._zeros((1,), cum_lengths.dtype)                  # same dtype as input (Parameter-safe)
        prev = self._concat0((zero, cum_lengths[:B - 1]))            # (B,)
        return cum_lengths - prev                                    # (B,)

    def _kv_from_cache_tnd(self, cache, block_tables, actual_seq_kvlen):
        """
        Build contiguous (T2, N_kv, D) from block-wise cache using block_tables and ragged kv lengths.

        cache:        (num_blocks, block_size, N_kv, D)
        block_tables: (B, max_blocks_per_seq) int32
        actual_seq_kvlen: cumulative kv lengths [B], last == T2
        """
        nb, bs, n_kv, d = cache.shape
        flat = self._reshape(cache, (nb * bs, n_kv, d))  # (nb*bs, N_kv, D)

        kv_cum  = actual_seq_kvlen
        kv_lens = self._diff_lengths(kv_cum)             # (B,)
        B       = kv_lens.shape[0]

        # positions [0..max_len-1] as int32 synthetic tensor
        max_len     = self._reduce_max(kv_lens)                      # scalar (same dtype as kv_lens)
        max_len_i32 = self._cast(max_len, mstype.int32)              # cast only the ReduceMax result (not a Parameter)
        start_i32   = ops.scalar_to_tensor(0, mstype.int32)
        step_i32    = ops.scalar_to_tensor(1, mstype.int32)
        pos_i32     = self._range(start_i32, max_len_i32, step_i32)  # (L,)

        # validity mask via broadcasting (no Tile)
        pos_for_mask = self._cast(self._expand_dims(pos_i32, 0), kv_lens.dtype)  # (1, L)
        kv_lens_2d   = self._expand_dims(kv_lens, 1)                              # (B, 1)
        valid_mask   = self._greater(kv_lens_2d, pos_for_mask)                    # (B, L) bool

        # block indexing
        bs_i32       = ops.scalar_to_tensor(bs, mstype.int32)
        blk_idx_row  = pos_i32 // bs_i32
        zeros_B1     = self._zeros((B, 1), mstype.int32)
        blk_idx      = zeros_B1 + blk_idx_row                                     # (B, L)

        table_i32 = self._cast(block_tables, mstype.int32)
        blk_ids   = self._gather_d(table_i32, 1, blk_idx)                         # (B, L)
        offsets   = pos_i32 - (blk_idx_row * bs_i32)
        offsets   = zeros_B1 + offsets                                            # (B, L)
        global_idx = blk_ids * bs_i32 + offsets                                   # (B, L)

        # ragged pack: select valid positions -> flatten to [T2]
        valid_idx_flat = self._masked_select(global_idx, valid_mask)              # (T2,)
        kv_tnd = self._gather(flat, valid_idx_flat, 0)                            # (T2, N_kv, D)
        return kv_tnd

    def _ensure_tnd_causal_mask(self, attn_mask, size=2048):
        """
        TND expects a fixed causal mask of shape (2048, 2048) with 1=discard, 0=keep (uint8/bool).
        If incoming mask matches, use it; otherwise build a canonical one.
        """
        if self._tnd_mask_2048 is None:
            ones_bool   = self._ones((size, size), mstype.bool_)
            upper_bool  = self._triu_strict(ones_bool)                  # strictly upper triangle
            self._tnd_mask_2048 = self._cast(upper_bool, mstype.uint8)  # 1=discard, 0=keep

        if attn_mask is not None:
            if len(attn_mask.shape) == 2 and attn_mask.shape[0] == size and attn_mask.shape[1] == size:
                if attn_mask.dtype == mstype.uint8 or attn_mask.dtype == mstype.bool_:
                    return attn_mask
        return self._tnd_mask_2048

    # -------------------- forward --------------------
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
        """Forward process of the FlashAttention."""

        # --- CRITICAL: cache write must match cache Parameter dtype ---
        # Only for the cache write we align K/V dtypes to cache dtypes to avoid Parameter conversion errors.
        if not self.use_multi_latent_attention:
            k_for_cache = key
            v_for_cache = value
            if key_cache is not None and hasattr(key_cache, "dtype"):
                if key is not None and key.dtype != key_cache.dtype:
                    k_for_cache = self._cast(key, key_cache.dtype)
            if value_cache is not None and hasattr(value_cache, "dtype"):
                if value is not None and value.dtype != value_cache.dtype:
                    v_for_cache = self._cast(value, value_cache.dtype)

            self.reshape_and_cache(k_for_cache, v_for_cache, key_cache, value_cache, slot_mapping)

        has_chunk = self._has_tnd_ragged(actual_seq_qlen, actual_seq_kvlen)
        self._dbg(
            f"enter: is_prefill={self.is_prefill}, has_chunk={has_chunk}, "
            f"Q={str(query.shape)}/{str(query.dtype)}, "
            f"K={str(key.shape) if key is not None else 'None'}/{str(key.dtype) if key is not None else 'None'}, "
            f"V={str(value.shape) if value is not None else 'None'}/{str(value.dtype) if value is not None else 'None'}"
        )

        if self.is_prefill:
            # ----- Prefill with TND FA (use provided K/V tensors) -----
            head_dim = self._infer_head_dim(query)
            q_tnd = self._to_tnd_from_th(query, self.head_num, head_dim)
            kv_heads = self.kv_head_num if self.kv_head_num else self.head_num
            k_tnd = self._to_tnd_from_th(key,   kv_heads, head_dim)
            v_tnd = self._to_tnd_from_th(value, kv_heads, head_dim)

            # Synthesize cumulative lengths if not provided (single sequence typical)
            if actual_seq_qlen is None:
                Tq = query.shape[0]
                actual_seq_qlen = self._zeros((1,), mstype.int32) + ops.scalar_to_tensor(Tq, mstype.int32)
            if actual_seq_kvlen is None:
                Tk = key.shape[0]
                actual_seq_kvlen = self._zeros((1,), mstype.int32) + ops.scalar_to_tensor(Tk, mstype.int32)

            attn_mask_tnd = self._ensure_tnd_causal_mask(attn_mask)
            self._dbg("branch: Prefill → TND_FA(prefill)")

            _, _, _, context = self.flash_attention_tnd_prefill(
                q_tnd, k_tnd, v_tnd,
                None, None,
                padding_mask, attn_mask_tnd,
                None,                          # prefix not used in TND
                actual_seq_qlen, actual_seq_kvlen
            )
            context = self._from_tnd_to_th(context)

        else:
            if has_chunk:
                # ----- 2D(+)-chunk prefill with TND FA (use K/V from cache) -----
                head_dim = self._infer_head_dim(query)
                q_tnd = self._to_tnd_from_th(query, self.head_num, head_dim)

                # Build packed K/V from cache based on block_tables + ragged kv lengths
                k_tnd = self._kv_from_cache_tnd(key_cache,   block_tables, actual_seq_kvlen)
                v_tnd = self._kv_from_cache_tnd(value_cache, block_tables, actual_seq_kvlen)
                attn_mask_tnd = self._ensure_tnd_causal_mask(attn_mask)
                self._dbg("branch: ChunkPrefill → TND_FA(chunk, from cache)")

                _, _, _, context = self.flash_attention_tnd_chunk(
                    q_tnd, k_tnd, v_tnd,
                    None, None,
                    padding_mask, attn_mask_tnd,
                    None,                          # prefix not used in TND
                    actual_seq_qlen, actual_seq_kvlen
                )
                context = self._from_tnd_to_th(context)

            else:
                # ----- Decode with PagedAttention -----
                self._dbg("branch: Decode → PagedAttention")
                if self.use_multi_latent_attention:
                    context = self.paged_attention(query, key_cache, key_cache,
                                                   block_tables, batch_valid_length, None,
                                                   None, attn_mask, q_seq_lens)
                else:
                    context = self.paged_attention(query, key_cache, value_cache,
                                                   block_tables, batch_valid_length, None,
                                                   None, attn_mask, q_seq_lens)

        self._dbg(f"exit: out={str(context.shape)}/{str(context.dtype)}")
        return context

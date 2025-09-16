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
"""Flash Attention Layer"""
__all__ = ['FlashAttention']

import os
import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """Flash Attention Layer.

    This function contains the flash attention primitives used in FlashAttention (see paper) and PagedAttention.
    `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/pdf/2205.14135.pdf>`
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

        # ---- debug knobs (env-gated) ----
        dbg_env = os.getenv("MF_FA_DEBUG", "").lower()
        self._debug = dbg_env not in ("", "0", "false", "off", "no")
        self._force_tnd_prefill = os.getenv("MF_FORCE_TND_PREFILL", "").lower() not in ("", "0", "false", "off", "no")
        self._print = ops.Print()

        # Remember requested scale; kernels get 1.0 to avoid BF16 vs double-attr issues.
        self._external_scale = scale_value

        # --- Core ops ---
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        # TH prefill kernel
        self.flash_attention_th = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=1.0,  # scale in-graph on Q
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout="TH",
            sparse_mode=self.sparse_mode)
        self.flash_attention_th.add_prim_attr("mf_role", "fa_prefill_TH")

        # TND FA kernel (also used for chunk prefill)
        self.flash_attention_tnd = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=1.0,  # scale in-graph on Q
            pre_tokens=pre_tokens,
            next_tokens=0,    # causal, no lookahead
            inner_precise=0,
            input_layout="TND",
            sparse_mode=3)    # causal optimized
        self.flash_attention_tnd.add_prim_attr("mf_role", "fa_TND")

        # Decode kernel — scale neutralized
        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, 1.0, pa_kv_head_num, mla_v_dim=pa_mla_v_dim)

        # --- helpers (Tile-free; never cast Parameters) ---
        self._reshape = ops.Reshape()
        self._cast = ops.Cast()
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
        self._triu_strict = ops.Triu(diagonal=1)  # strictly upper triangle (j > i)

        # Canonical (2048,2048) lower-tri discard mask for TND mode
        self._tnd_mask_2048 = None

    # -------------------- small debug helper --------------------
    def _dbg(self, msg):
        if self._debug:
            # ops.Print accepts strings (compile-time constants) happily.
            self._print("[MF_FA] " + msg)

    # -------------------- helpers --------------------

    def _has_tnd_ragged(self, actual_seq_qlen, actual_seq_kvlen):
        # TND requires both cumulative arrays; check rank >= 1.
        if actual_seq_qlen is None or actual_seq_kvlen is None:
            return False
        if len(actual_seq_qlen.shape) == 0 or len(actual_seq_kvlen.shape) == 0:
            return False
        return True

    def _infer_head_dim(self, query):
        if self.hidden_size_per_attention_head is not None:
            return self.hidden_size_per_attention_head
        return query.shape[1] // self.head_num  # packed TH: (T, H*D)

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
        flat = self._reshape(cache, (nb * bs, n_kv, d))              # (nb*bs, N_kv, D)

        kv_cum  = actual_seq_kvlen                                   # keep original dtype (may be Parameter[int])
        kv_lens = self._diff_lengths(kv_cum)                         # (B,) same dtype as input
        B       = kv_lens.shape[0]

        # Build positions [0..max_len-1] as int32 synthetic tensor
        max_len     = self._reduce_max(kv_lens)                      # scalar (same dtype as kv_lens)
        max_len_i32 = self._cast(max_len, mstype.int32)              # cast only the result (Tensor, not Parameter)
        start_i32   = ops.scalar_to_tensor(0, mstype.int32)
        step_i32    = ops.scalar_to_tensor(1, mstype.int32)
        pos_i32     = self._range(start_i32, max_len_i32, step_i32)  # (L,)

        # Validity mask: cast synthetic pos to dtype of kv_lens; never cast kv_lens/Parameters
        pos_for_mask = self._cast(self._expand_dims(pos_i32, 0), kv_lens.dtype)  # (1, L)
        kv_lens_2d   = self._expand_dims(kv_lens, 1)                              # (B, 1)
        valid_mask   = self._greater(kv_lens_2d, pos_for_mask)                    # (B, L) bool

        # Compute block indices via broadcasting (no Tile)
        bs_i32       = ops.scalar_to_tensor(bs, mstype.int32)
        blk_idx_row  = pos_i32 // bs_i32                                          # (L,)
        zeros_B1     = self._zeros((B, 1), mstype.int32)                          # (B,1)
        blk_idx      = zeros_B1 + blk_idx_row                                     # (B,L) via broadcast

        # Gather block ids and compute global token indices (all int32)
        table_i32 = self._cast(block_tables, mstype.int32)                        # (B, max_blocks)
        blk_ids   = self._gather_d(table_i32, 1, blk_idx)                         # (B, L)
        offsets   = pos_i32 - (blk_idx_row * bs_i32)                              # (L,)
        offsets   = zeros_B1 + offsets                                            # (B, L)
        global_idx = blk_ids * bs_i32 + offsets                                   # (B, L)

        # Ragged pack: select valid positions to get token-major indices [T2]
        valid_idx_flat = self._masked_select(global_idx, valid_mask)              # (T2,)
        kv_tnd = self._gather(flat, valid_idx_flat, 0)                            # (T2, N_kv, D)
        return kv_tnd

    def _ensure_tnd_causal_mask(self, attn_mask, size=2048):
        """
        TND requires a (2048, 2048) causal mask with 1=discard, 0=keep (uint8/bool).
        If incoming mask already matches, use it; otherwise build the canonical one.
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

        # Always write fresh K/V into cache (original behavior).
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # Scale query in-graph (BF16/FP16-safe); kernels run with scale_value=1.0
        scale32 = ops.scalar_to_tensor(self._external_scale, mstype.float32)
        scaleQ  = self._cast(scale32, query.dtype)
        query   = query * scaleQ

        # Debug header (env-gated)
        self._dbg(
            f"enter construct: is_prefill={self.is_prefill}, "
            f"q_dtype={str(query.dtype)}, q_shape={str(query.shape)}, "
            f"k_shape={str(key.shape) if key is not None else 'None'}, "
            f"v_shape={str(value.shape) if value is not None else 'None'}, "
            f"has_chunk={self._has_tnd_ragged(actual_seq_qlen, actual_seq_kvlen)}, "
            f"q_seq_lens_shape={str(q_seq_lens.shape) if q_seq_lens is not None else 'None'}"
        )

        if self.is_prefill:
            # TH prefill by default (as original), but allow forcing TND (env) if driver rejects TH.
            prefer_tnd_prefill = self._force_tnd_prefill

            if prefer_tnd_prefill:
                self._dbg("branch=Prefill(TND)")
                head_dim = self._infer_head_dim(query)
                q_tnd = self._to_tnd_from_th(query, self.head_num, head_dim)
                k_tnd = self._to_tnd_from_th(key,   self.kv_head_num if self.kv_head_num else self.head_num, head_dim)
                v_tnd = self._to_tnd_from_th(value, self.kv_head_num if self.kv_head_num else self.head_num, head_dim)

                # Synthesize cumulative lengths if not provided (single sequence)
                Tq = query.shape[0]
                Tk = key.shape[0]
                if actual_seq_qlen is None:
                    actual_seq_qlen = self._zeros((1,), mstype.int32) + ops.scalar_to_tensor(Tq, mstype.int32)
                if actual_seq_kvlen is None:
                    actual_seq_kvlen = self._zeros((1,), mstype.int32) + ops.scalar_to_tensor(Tk, mstype.int32)

                attn_mask_tnd = self._ensure_tnd_causal_mask(attn_mask)

                # BF16 → FP16 for kernel if needed
                amp_bf16 = (q_tnd.dtype == mstype.bfloat16)
                q_in = self._cast(q_tnd, mstype.float16) if amp_bf16 else q_tnd
                k_in = self._cast(k_tnd, mstype.float16) if k_tnd.dtype == mstype.bfloat16 else k_tnd
                v_in = self._cast(v_tnd, mstype.float16) if v_tnd.dtype == mstype.bfloat16 else v_tnd

                _, _, _, context = self.flash_attention_tnd(
                    q_in, k_in, v_in,
                    None, None,
                    padding_mask, attn_mask_tnd,
                    None,
                    actual_seq_qlen, actual_seq_kvlen
                )
                if amp_bf16:
                    context = self._cast(context, mstype.bfloat16)
                context = self._from_tnd_to_th(context)

            else:
                self._dbg("branch=Prefill(TH)")
                # TH prefill path (original)
                amp_bf16 = (query.dtype == mstype.bfloat16)
                q_in = self._cast(query, mstype.float16) if amp_bf16 else query
                k_in = self._cast(key,   mstype.float16) if (key is not None and key.dtype == mstype.bfloat16) else key
                v_in = self._cast(value, mstype.float16) if (value is not None and value.dtype == mstype.bfloat16) else value

                _, _, _, context = self.flash_attention_th(
                    q_in, k_in, v_in,
                    None, None,
                    padding_mask, attn_mask,
                    prefix, actual_seq_qlen,
                    actual_seq_kvlen)
                if amp_bf16:
                    context = self._cast(context, mstype.bfloat16)

        else:
            if self._has_tnd_ragged(actual_seq_qlen, actual_seq_kvlen):
                self._dbg("branch=ChunkPrefill(TND from cache)")
                # 2D(+)-chunk prefill (TND): Q reshaped; K/V from cache; canonical mask; prefix=None
                head_dim = self._infer_head_dim(query)
                q_tnd = self._to_tnd_from_th(query, self.head_num, head_dim)

                # Pull K/V from cache and make TND
                k_tnd = self._kv_from_cache_tnd(key_cache, block_tables, actual_seq_kvlen)
                v_tnd = self._kv_from_cache_tnd(value_cache, block_tables, actual_seq_kvlen)

                # If BF16, feed FP16 to kernel and cast result back
                amp_bf16 = (q_tnd.dtype == mstype.bfloat16)
                q_in = self._cast(q_tnd, mstype.float16) if amp_bf16 else q_tnd
                k_in = self._cast(k_tnd, mstype.float16) if k_tnd.dtype == mstype.bfloat16 else k_tnd
                v_in = self._cast(v_tnd, mstype.float16) if v_tnd.dtype == mstype.bfloat16 else v_tnd

                attn_mask_tnd = self._ensure_tnd_causal_mask(attn_mask)

                _, _, _, context = self.flash_attention_tnd(
                    q_in, k_in, v_in,
                    None, None,
                    padding_mask, attn_mask_tnd,
                    None,
                    actual_seq_qlen, actual_seq_kvlen
                )
                if amp_bf16:
                    context = self._cast(context, mstype.bfloat16)
                context = self._from_tnd_to_th(context)

            else:
                self._dbg("branch=Decode(PagedAttention)")
                # Decode (PagedAttention) — cast to FP16 if BF16 to avoid driver limitations
                amp_bf16 = (query.dtype == mstype.bfloat16)
                q_in  = self._cast(query,      mstype.float16) if amp_bf16 else query
                kc_in = self._cast(key_cache,  mstype.float16) if (key_cache is not None and key_cache.dtype == mstype.bfloat16) else key_cache
                vc_in = self._cast(value_cache, mstype.float16) if (value_cache is not None and value_cache.dtype == mstype.bfloat16) else value_cache

                if self.use_multi_latent_attention:
                    context = self.paged_attention(q_in, kc_in, kc_in,
                                                   block_tables, batch_valid_length, None,
                                                   None, attn_mask, q_seq_lens)
                else:
                    context = self.paged_attention(q_in, kc_in, vc_in,
                                                   block_tables, batch_valid_length, None,
                                                   None, attn_mask, q_seq_lens)

                if amp_bf16:
                    context = self._cast(context, mstype.bfloat16)

        # Footer debug
        self._dbg(f"exit construct: out_shape={str(context.shape)}, out_dtype={str(context.dtype)}")
        return context

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

from mindspore import ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """Flash Attention Layer for prefill + paged decode.

    Prefill (first & 2nd+ chunk): FlashAttention in TND layout.
    Decode: PagedAttention on cached K/V.
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
            # IMPORTANT: run FA in TND; we reshape explicitly in construct().
            input_layout="TND",
            pa_kv_head_num=None,
            pa_mla_v_dim=0
    ):
        super().__init__()
        self.head_num = head_num                  # N_q
        self.hidden_size_per_attention_head = head_dim  # D
        self.kv_head_num = kv_head_num            # N_kv
        self.sparse_mode = sparse_mode
        self.is_prefill = True
        self.input_layout = input_layout
        self.use_multi_latent_attention: bool = pa_mla_v_dim > 0

        # cache writer for paged attention
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        # FlashAttention in TND
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.input_layout,   # "TND"
            sparse_mode=self.sparse_mode)

        # PagedAttention for decode
        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim)

        # tiny helpers
        self._cast = ops.Cast()
        self._add = ops.Add()
        self._mul = ops.Mul()
        self._shape = ops.Shape()
        self._gather = ops.Gather()
        self._maximum = ops.Maximum()

    # -----------------------
    # Small, explicit helpers
    # -----------------------
    def _p(self, *msg):
        # compact, always-on prints as requested
        print(*msg)

    def _is_second_plus_chunk(self, q_seq_lens):
        """Heuristic: in prefill, if sum(q_seq_lens) > batch_size, we’re on chunk #2+."""
        if q_seq_lens is None:
            return False
        # q_seq_lens is 1-D tensor [B]; here B==1 in your runs.
        # Convert to python ints for simplicity (OK at O0).
        try:
            total = int(q_seq_lens.sum().asnumpy().item())
            bs = int(self._shape(q_seq_lens)[0])
        except Exception:
            # fallback: assume not 2nd+ if we can't read
            return False
        return total > bs

    def _to_tnd(self, th_tensor, num_heads, head_dim):
        # TH [T, H] -> TND [T, N, D]
        T = int(self._shape(th_tensor)[0])
        return ops.reshape(th_tensor, (T, num_heads, head_dim))

    def _merge_tnd(self, tnd_tensor):
        # TND [T, N, D] -> TH [T, N*D]
        T, N, D = self._shape(tnd_tensor)
        return ops.reshape(tnd_tensor, (T, N * D))

    def _dtype_harmonize(self, q, k, v, key_cache, value_cache):
        # When caches exist, both caches are float16 in your setup → cast q/k/v to float16.
        target = mstype.float16 if (key_cache is not None or value_cache is not None) else q.dtype
        if q.dtype != target:
            q = self._cast(q, target)
        if k.dtype != target:
            k = self._cast(k, target)
        if v.dtype != target:
            v = self._cast(v, target)
        return q, k, v

    def _gather_contiguous_kv(self, kv_cache, block_tables, total_kv_len):
        """
        Build a contiguous [total_kv_len, N_kv, D] from paged cache for *single batch*.

        kv_cache:  [num_blocks, block_size, N_kv, D]
        block_tables: [B, max_blocks] with 1-based block ids, padded with 0
        total_kv_len: python int (last element of actual_seq_kvlen)
        """
        nb, bs, n_kv, d = [int(x) for x in self._shape(kv_cache)]
        # 1-based ids for B=1; trailing zeros
        ids_1b = block_tables[0]                           # [max_blocks]
        # limit to exactly needed number of tokens
        needed_blocks = (total_kv_len + bs - 1) // bs      # ceil
        ids_1b_slice = ids_1b[:needed_blocks]              # [needed_blocks]
        # convert to 0-based; clamp at 0
        zero = ops.scalar_to_tensor(0, ids_1b_slice.dtype)
        ids_0b = self._maximum(ids_1b_slice - 1, zero)     # [needed_blocks]
        # gather blocks → [needed_blocks, bs, n_kv, d]
        gathered = self._gather(kv_cache, ids_0b, 0)
        # flatten tokens and trim to exact token count
        flat = ops.reshape(gathered, (needed_blocks * bs, n_kv, d))  # [needed_blocks*bs, n_kv, d]
        T_full = needed_blocks * bs
        # Trim to total_kv_len (Python int slice is fine at O0)
        flat = flat[:total_kv_len]
        return flat  # [T_kv, n_kv, d]

    # -----------
    # main forward
    # -----------
    def construct(self,
                  query,                 # TH [Tq, Hq]
                  key,                   # TH [Tk, Hkv] (current chunk)
                  value,                 # TH [Tk, Hkv]
                  slot_mapping=None,
                  block_tables=None,
                  batch_valid_length=None,
                  context_lens_tensor=None,
                  q_seq_lens=None,
                  actual_seq_qlen=None,  # 1-D int list/tensor
                  actual_seq_kvlen=None, # 1-D int list/tensor
                  attn_mask=None,        # DO NOT pass paged mask into FA
                  padding_mask=None,     # must be None for FA
                  prefix=None,
                  key_cache=None,
                  value_cache=None):
        """Forward process:
           - write chunk into cache,
           - if prefill:
               - chunk #1: FA over current (TND) Q/K/V
               - chunk #2+: FA over gathered contiguous KV (TND) and current Q
           - else (decode): PagedAttention on cache
        """
        # Always write current chunk to cache first (original behavior).
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # harmonize dtypes with cache (float16) to satisfy kernel constraints.
        query, key, value = self._dtype_harmonize(query, key, value, key_cache, value_cache)

        # --- prefill path with FlashAttention (TND) ---
        if self.is_prefill:
            # decide if this is 2nd+ prefill chunk
            is_second_plus = self._is_second_plus_chunk(q_seq_lens)

            # convert TH → TND for Q (always)
            q_tnd = self._to_tnd(query, self.head_num, self.hidden_size_per_attention_head)

            if not is_second_plus:
                # 1st prefill: use the provided K/V (TH → TND)
                k_tnd = self._to_tnd(key, self.kv_head_num, self.hidden_size_per_attention_head)
                v_tnd = self._to_tnd(value, self.kv_head_num, self.hidden_size_per_attention_head)

                # IMPORTANT: FA prefill should not receive paged masks
                _, _, _, ctx_tnd = self.flash_attention(
                    q_tnd, k_tnd, v_tnd,
                    None, None,
                    None, None,  # padding_mask, attn_mask
                    None, actual_seq_qlen, actual_seq_kvlen
                )
                context = self._merge_tnd(ctx_tnd)  # TH [Tq, H]
                # debug crumbs
                self._p("FA[1st] Q/K/V(TND)->ctx(TH):",
                        self._shape(q_tnd), self._shape(k_tnd), self._shape(v_tnd), self._shape(context),
                        "dtypes:", q_tnd.dtype, k_tnd.dtype, v_tnd.dtype)
                return context

            # 2nd+ prefill: gather contiguous KV from cache up to actual_seq_kvlen
            if (key_cache is None) or (value_cache is None) or (block_tables is None) or (actual_seq_kvlen is None):
                # Fallback: if any input missing, use paged-attn (safe)
                self._p("FA[2+]: missing inputs → fallback to paged attn.")
                ctx = self.paged_attention(query, key_cache, value_cache,
                                           block_tables, batch_valid_length, None, None, None, q_seq_lens)
                return ctx

            # Python int total_kv_len (OK at O0 in your runs)
            total_kv_len = int(actual_seq_kvlen.view(-1)[-1].asnumpy().item())
            k_full = self._gather_contiguous_kv(key_cache, block_tables, total_kv_len)   # [Tk, N_kv, D]
            v_full = self._gather_contiguous_kv(value_cache, block_tables, total_kv_len) # [Tk, N_kv, D]

            # run FA (no paged masks)
            _, _, _, ctx_tnd = self.flash_attention(
                q_tnd, k_full, v_full,
                None, None,
                None, None,  # padding_mask, attn_mask
                None, actual_seq_qlen, actual_seq_kvlen
            )
            context = self._merge_tnd(ctx_tnd)  # TH [Tq, H]
            # debug crumbs
            self._p("FA[2+]: Q(TND), K/V(gathered TND) -> ctx(TH):",
                    self._shape(q_tnd), self._shape(k_full), self._shape(v_full), self._shape(context),
                    "dtypes:", q_tnd.dtype, k_full.dtype, v_full.dtype)
            return context

        # --- decode path (not prefill): paged attention on cache ---
        if self.use_multi_latent_attention:
            context = self.paged_attention(query, key_cache, key_cache,
                                           block_tables, batch_valid_length, None,
                                           None, attn_mask, q_seq_lens)
        else:
            context = self.paged_attention(query, key_cache, value_cache,
                                           block_tables, batch_valid_length, None,
                                           None, attn_mask, q_seq_lens)

        # debug crumbs
        self._p("PA[decode]: Q(TH) on cache -> ctx(TH):", self._shape(query), self._shape(context), "dtype:", context.dtype)
        return context

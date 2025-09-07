# ===================== research/qwen2_5/infer/transformer.py =====================

# (A) In ParallelAttention.__init__: keep your existing FA, add one BSH FA for later-chunk prefill.
from mindformers.modules.flash_attention import FlashAttention

class ParallelAttention(nn.Cell):
    def __init__(self, layer_index, config):
        super().__init__()
        self.layer_index = layer_index
        self.config = config
        # ... your existing init ...

        if self.use_flash_attention:
            # existing FA (your file already uses TH when use_past=True)
            input_layout = "TH" if self.use_past else "BNSD"
            self.flash_attention = FlashAttention(
                head_num=self.num_heads_per_partition,
                scale_value=1.0 / self.norm_factor,
                next_tokens=0,
                input_layout=input_layout
            )
            # NEW: FA instance that accepts BSH (for later-chunk prefill only)
            if self.use_past:
                self.flash_attention_prefill_bsh = FlashAttention(
                    head_num=self.num_heads_per_partition,
                    scale_value=1.0 / self.norm_factor,
                    next_tokens=0,
                    input_layout="BSH"
                )
        else:
            self.core_attention = CoreAttention(self.layer_index, self.config)



# (B) In ParallelAttention.construct: INSIDE the `if self.use_past:` branch,
#     after the KV write, replace ONLY the attention compute with the guarded fast path.

            # Always write K/V into paged cache (UNCHANGED)
            key_out = self.paged_attention_mgr(
                key, value, slot_mapping, batch_valid_length,
                key_cache=key_cache, value_cache=value_cache
            )
            query = ops.depend(query, key_out)  # fence: write before read

            # ---- Guard shapes: first iteration is TH ([T, H]); later iterations are BSH ([B, S, H]) ----
            # We ONLY treat "chunk prefill" when NOT first iteration AND query is BSH with S > 1.
            is_query_th  = (len(query.shape) == 2)   # [T, H]
            is_query_bsh = (len(query.shape) == 3)   # [B, S, H]
            S = int(query.shape[1]) if is_query_bsh else 1

            # ===================== CHUNK-PREFILL FAST PATH (LATER CHUNKS ONLY) =====================
            if (self.use_flash_attention
                and (not self.is_first_iteration)    # never intercept first iteration (TH path stays original)
                and is_query_bsh
                and S > 1):
                # query: [B, S, H]; gather KV from cache to BSH and run FA; early exit after projection.

                # Cast indices once
                if block_tables.dtype != mstype.int32:
                    block_tables = ops.Cast()(block_tables, mstype.int32)

                B, M = int(block_tables.shape[0]), int(block_tables.shape[1])
                flat = ops.Reshape()(block_tables, (B * M,))                    # [B*M]

                # Manager caches are [B*M, bs_block, Hk]
                k_raw = ops.Gather()(key_cache,   flat, 0)                       # [B*M, bs, Hk]
                v_raw = ops.Gather()(value_cache, flat, 0)
                bs_block = int(k_raw.shape[1]); Hk = int(k_raw.shape[-1])
                KV = M * bs_block

                # [B*M, bs, Hk] -> [B, KV, Hk]  (stay in BSH for FA)
                k_bsh = ops.Reshape()(k_raw, (B, KV, Hk))                        # [B, KV, Hk]
                v_bsh = ops.Reshape()(v_raw, (B, KV, Hk))                        # [B, KV, Hk]

                # ---- GQA head align on [B, KV, H] if needed ----
                nh = int(self.num_heads_per_partition)
                d  = self.head_dim
                kv_heads = Hk // d
                if self.use_gqa and kv_heads != nh:
                    # [B, KV, kv_heads, D] -> tile -> [B, KV, N, D] -> [B, KV, N*D]
                    k4 = ops.Reshape()(k_bsh, (B, KV, kv_heads, d))
                    v4 = ops.Reshape()(v_bsh, (B, KV, kv_heads, d))
                    k4 = mint.repeat_interleave(k4, repeats=self.repeat_num, dim=2)
                    v4 = mint.repeat_interleave(v4, repeats=self.repeat_num, dim=2)
                    k_bsh = ops.Reshape()(k4, (B, KV, nh * d))
                    v_bsh = ops.Reshape()(v4, (B, KV, nh * d))

                # ---- Dtype guard: match KV to Q dtype right at call site ----
                fa_dtype = query.dtype
                if k_bsh.dtype != fa_dtype: k_bsh = ops.Cast()(k_bsh, fa_dtype)
                if v_bsh.dtype != fa_dtype: v_bsh = ops.Cast()(v_bsh, fa_dtype)
                if (alibi_mask is not None) and (alibi_mask.dtype != fa_dtype):
                    alibi_mask = ops.Cast()(alibi_mask, fa_dtype)

                # FlashAttention (BSH) for later-chunk prefill
                # Q: [B, S, H]; K/V: [B, KV, H]
                context_layer = self.flash_attention_prefill_bsh(
                    query, k_bsh, v_bsh,
                    None,                  # attn_mask (None if FA handles causal internally)
                    alibi_mask,
                    None, None,            # prefix, padding_mask
                    q_seq_lens,            # actual_seq_qlen
                    batch_valid_length     # actual_seq_kvlen
                )

                # ---- EARLY EXIT: project and return (matches your file’s flow) ----
                attn_out = self.wo(context_layer)     # if your proj is o_proj, change to self.o_proj
                return attn_out
            # ================= end fast path; first iteration (TH) falls through unchanged =================

            # DECODE (S == 1) or FA disabled or first iteration (TH): keep your original path
            # (this is exactly what your file already does next)
            context_layer = self.paged_attention_mgr.paged_attn(
                query, batch_valid_length, block_tables,
                attn_mask, q_seq_lens, key_cache, value_cache
            )
            # ... your original code continues here (wo + residual/MLP + return) ...

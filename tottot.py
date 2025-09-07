# ===================== research/qwen2_5/infer/transformer.py =====================

# --- in class ParallelAttention.__init__ (add the BSH FA; keep your existing code) ---
from mindformers.modules.flash_attention import FlashAttention

class ParallelAttention(nn.Cell):
    def __init__(self, layer_index, config):
        super().__init__()
        self.layer_index = layer_index
        self.config = config
        # ... your existing init code ...

        # existing FA (your file likely builds one already)
        if self.use_flash_attention:
            input_layout = "TH" if self.use_past else "BNSD"
            self.flash_attention = FlashAttention(
                head_num=self.num_heads_per_partition,
                scale_value=1.0 / self.norm_factor,
                next_tokens=0,
                input_layout=input_layout
            )
            # NEW: a dedicated FA for chunk-prefill that accepts BSH directly
            if self.use_past:
                self.flash_attention_prefill_bsh = FlashAttention(
                    head_num=self.num_heads_per_partition,
                    scale_value=1.0 / self.norm_factor,
                    next_tokens=0,
                    input_layout="BSH",
                    # optional knobs if available in your FA build:
                    # sparse_mode=2,            # causal
                    # use_attention_mask=False,
                    # use_alibi_mask=getattr(self.config, "use_alibi", False),
                    # use_actual_seqlen=True,
                )
        else:
            self.core_attention = CoreAttention(self.layer_index, self.config)

# --- in class ParallelAttention.construct, inside `if self.use_past:` block ---
# Keep everything above intact until *after* the KV write. Replace only the
# attention compute with the early-exit prefill fast path below.

            # Always write K/V into paged cache (UNCHANGED)
            key_out = self.paged_attention_mgr(
                key, value, slot_mapping, batch_valid_length,
                key_cache=key_cache, value_cache=value_cache
            )
            query = ops.depend(query, key_out)

            # x is [B, S, H_total] in your file; use it to read S robustly
            bs, seq_len, _ = x.shape

            # ======== EARLY-EXIT FAST PATH for CHUNK PREFILL (S > 1) ========
            if self.use_flash_attention and seq_len > 1:
                # Gather ALL blocks once; FA will clamp via actual lengths
                if block_tables.dtype != mstype.int32:
                    block_tables = ops.Cast()(block_tables, mstype.int32)
                B, M = int(block_tables.shape[0]), int(block_tables.shape[1])
                flat = ops.Reshape()(block_tables, (B * M,))                     # [B*M]

                # key_cache/value_cache layout from manager: [B*M, bs_block, Hk]
                k_raw = ops.Gather()(key_cache,   flat, 0)                        # [B*M, bs, Hk]
                v_raw = ops.Gather()(value_cache, flat, 0)
                bs_block = int(k_raw.shape[1])
                Hk = int(k_raw.shape[-1])
                KV = M * bs_block

                # [B*M, bs, Hk] -> [B, KV, Hk]  (BSH layout for FA)
                k_bsh = ops.Reshape()(k_raw, (B, KV, Hk))                         # [B, KV, Hk]
                v_bsh = ops.Reshape()(v_raw, (B, KV, Hk))                         # [B, KV, Hk]

                # ---- GQA head align (only if needed) ----
                # We want K/V hidden to match Q hidden: H = N*D with N=self.num_heads_per_partition
                nh = int(self.num_heads_per_partition)
                d  = self.head_dim
                kv_heads = Hk // d
                if self.use_gqa and kv_heads != nh:
                    # [B, KV, kv_heads, D] -> tile heads -> [B, KV, nh, D] -> [B, KV, nh*D]
                    k4 = ops.Reshape()(k_bsh, (B, KV, kv_heads, d))
                    v4 = ops.Reshape()(v_bsh, (B, KV, kv_heads, d))
                    k4 = mint.repeat_interleave(k4, repeats=self.repeat_num, dim=2)
                    v4 = mint.repeat_interleave(v4, repeats=self.repeat_num, dim=2)
                    k_bsh = ops.Reshape()(k4, (B, KV, nh * d))
                    v_bsh = ops.Reshape()(v4, (B, KV, nh * d))

                # ---- Dtype guard: match K/V to Q dtype right at the call site ----
                fa_dtype = query.dtype                 # query is already [B, S, H] (BSH)
                if k_bsh.dtype != fa_dtype: k_bsh = ops.Cast()(k_bsh, fa_dtype)
                if v_bsh.dtype != fa_dtype: v_bsh = ops.Cast()(v_bsh, fa_dtype)
                if (alibi_mask is not None) and (alibi_mask.dtype != fa_dtype):
                    alibi_mask = ops.Cast()(alibi_mask, fa_dtype)

                # ---- FlashAttention (BSH) over prefill; use provided lengths ----
                # Shapes (your file’s comments):
                #   Q: [B, S, H]   (BSH)
                #   K: [B, KV, H]  (BSH)
                #   V: [B, KV, H]  (BSH)
                context_layer = self.flash_attention_prefill_bsh(
                    query, k_bsh, v_bsh,
                    None,                    # attn_mask: None if FA set causal internally; else pass attn_mask
                    alibi_mask,              # keep if your model uses ALiBi
                    None, None,              # prefix, padding_mask (unused)
                    q_seq_lens,              # actual_seq_qlen
                    batch_valid_length       # actual_seq_kvlen
                )
                # context_layer: [B, S, H]  -> project and EARLY RETURN
                attn_out = self.wo(context_layer)      # use self.o_proj if that’s the name in your file
                return attn_out
            # ====================== end chunk-prefill fast path ======================

            # DECODE (S == 1) or FA disabled: keep your original paged-attention path
            context_layer = self.paged_attention_mgr.paged_attn(
                query, batch_valid_length, block_tables,
                attn_mask, q_seq_lens, key_cache, value_cache
            )
            # ... your original code continues (wo + residual/MLP) ...

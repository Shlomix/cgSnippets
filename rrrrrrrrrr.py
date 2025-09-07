# ---- at the very top of the `if self.use_past:` branch in construct(), keep your existing: ----
# comp_dt = getattr(self, "compute_dtype", None) or getattr(self.config, "compute_dtype", None) or x.dtype
# if x.dtype != comp_dt: x = ops.Cast()(x, comp_dt)

# ---- after KV write (unchanged) ----
key_out = self.paged_attention_mgr(
    key, value, slot_mapping, batch_valid_length,
    key_cache=key_cache, value_cache=value_cache
)
query = ops.depend(query, key_out)

# common FA dtype
fa_dtype = (getattr(self, "compute_dtype", None)
            or getattr(self.config, "compute_dtype", None)
            or query.dtype)

# robust chunk length from x even when x is TH:
if len(x.shape) == 3:   # [B, S, H]
    B = int(x.shape[0])
    S_cur = int(x.shape[1])
else:                   # TH: [T, H]
    B = int(block_tables.shape[0])  # batch from tables/valid_length
    S_cur = int(x.shape[0]) if B == 1 else int(x.shape[0]) // max(B, 1)

# ===================== LATER-CHUNK CHUNK-PREFILL FAST PATH (TH) =====================
if self.use_flash_attention and (not self.is_first_iteration) and (S_cur > 1):
    # Ensure Q is TH
    if len(query.shape) == 3:  # BSH -> TH
        Hq = int(query.shape[-1])
        query = ops.Reshape()(query, (B * S_cur, Hq))   # [Tq, H]

    # ---- Gather KV from paged cache ----
    if block_tables.dtype != mstype.int32:
        block_tables = ops.Cast()(block_tables, mstype.int32)
    M = int(block_tables.shape[1])
    flat = ops.Reshape()(block_tables, (B * M,))           # [B*M]

    # caches: [B*M, bs_block, Hk]
    k_raw = ops.Gather()(key_cache,   flat, 0)             # [B*M, bs, Hk]
    v_raw = ops.Gather()(value_cache, flat, 0)
    bs_blk = int(k_raw.shape[1]); Hk = int(k_raw.shape[-1])
    KV = M * bs_blk

    # reshape to BSH to perform GQA align safely
    k_bsh = ops.Reshape()(k_raw, (B, KV, Hk))              # [B, KV, Hk]
    v_bsh = ops.Reshape()(v_raw, (B, KV, Hk))              # [B, KV, Hk]

    # ---- GQA head align (only if needed) on BSH ----
    nh = int(self.num_heads_per_partition)
    d  = self.head_dim
    kv_heads = Hk // d
    if self.use_gqa and kv_heads != nh:
        # [B, KV, kv_heads, D] -> tile -> [B, KV, nh, D] -> [B, KV, nh*D]
        k4 = ops.Reshape()(k_bsh, (B, KV, kv_heads, d))
        v4 = ops.Reshape()(v_bsh, (B, KV, kv_heads, d))
        k4 = mint.repeat_interleave(k4, repeats=self.repeat_num, dim=2)
        v4 = mint.repeat_interleave(v4, repeats=self.repeat_num, dim=2)
        k_bsh = ops.Reshape()(k4, (B, KV, nh * d))
        v_bsh = ops.Reshape()(v4, (B, KV, nh * d))

    # ---- BSH -> TH for KV to match FA(TH) input layout ----
    Hq = int(query.shape[-1])                # hidden after GQA align
    k_th = ops.Reshape()(k_bsh, (B * KV, Hq))   # [Tk, H]
    v_th = ops.Reshape()(v_bsh, (B * KV, Hq))   # [Tk, H]

    # ---- Cast FA inputs to compute dtype (BF16) ----
    if query.dtype != fa_dtype: query = ops.Cast()(query, fa_dtype)
    if k_th.dtype  != fa_dtype: k_th  = ops.Cast()(k_th,  fa_dtype)
    if v_th.dtype  != fa_dtype: v_th  = ops.Cast()(v_th,  fa_dtype)
    if (alibi_mask is not None) and (alibi_mask.dtype != fa_dtype):
        alibi_mask = ops.Cast()(alibi_mask, fa_dtype)

    # ---- FlashAttention (TH) with actual sequence lengths ----
    # Q: [B*S_cur, H],  K/V: [B*KV, H]
    context_th = self.flash_attention(
        query, k_th, v_th,
        attn_mask,            # or None if your FA is configured as causal internally
        alibi_mask,
        None, None,
        q_seq_lens,           # actual_seq_qlen  (vector length B)
        batch_valid_length    # actual_seq_kvlen (vector length B)
    )

    # ---- TH -> BSH for projection and EARLY RETURN ----
    context_layer = ops.Reshape()(context_th, (B, S_cur, Hq))   # [B, S, H]
    attn_out = self.wo(context_layer)   # change to self.o_proj if that’s your name
    return attn_out
# ===================== END LATER-CHUNK FAST PATH =====================

# First-iteration TH FA branch and decode branch remain exactly as you have them,
# with the BF16 casts you already added before calling self.flash_attention or paged_attn.

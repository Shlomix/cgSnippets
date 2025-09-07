# ========================= PATCH START (TH-only) =========================
# --- inside ParallelAttention.construct(...), in the `if self.use_past:` branch ---

# 0) NEW: stabilize the graph input signature (avoid BF16/FP16 mismatch)
comp_dt = (getattr(self, "compute_dtype", None)
           or getattr(self.config, "compute_dtype", None)
           or x.dtype)
if x.dtype != comp_dt:
    x = ops.Cast()(x, comp_dt)

# ... your existing code that computes q/k/v (query, key, value) ...

# 1) KV write (UNCHANGED)
key_out = self.paged_attention_mgr(
    key, value, slot_mapping, batch_valid_length,
    key_cache=key_cache, value_cache=value_cache
)
query = ops.depend(query, key_out)  # ensure writes happen before any read

# 2) Common dtype for FA inputs
fa_dtype = comp_dt

# 3) Robust chunk length (works even when x/query are TH)
if len(x.shape) == 3:      # x: [B, S, H]
    B = int(x.shape[0])
    S_cur = int(x.shape[1])
else:                      # x: [T, H]
    B = int(block_tables.shape[0])
    S_cur = int(x.shape[0]) if B == 1 else int(x.shape[0]) // max(B, 1)

# ===================== LATER-CHUNK CHUNK-PREFILL FAST PATH (TH) =====================
# We ONLY intercept later-chunk prefill (not first iteration), and only when S_cur > 1
if self.use_flash_attention and (not self.is_first_iteration) and (S_cur > 1):
    # 3a) Ensure Q is TH: [B,S,H] -> [Tq,H] if needed
    if len(query.shape) == 3:
        Hq = int(query.shape[-1])
        query = ops.Reshape()(query, (B * S_cur, Hq))         # [Tq, H]
    else:
        Hq = int(query.shape[-1])                             # already [Tq, H]

    # 3b) Gather K/V from the paged cache and FLATTEN to TH directly
    if block_tables.dtype != mstype.int32:
        block_tables = ops.Cast()(block_tables, mstype.int32)
    M = int(block_tables.shape[1])
    flat = ops.Reshape()(block_tables, (B * M,))              # [B*M]

    # caches shape: [B*M, bs_block, Hk]
    k_raw = ops.Gather()(key_cache,   flat, 0)                # [B*M, bs, Hk]
    v_raw = ops.Gather()(value_cache, flat, 0)
    bs_blk = int(k_raw.shape[1])
    Hk = int(k_raw.shape[-1])
    KV = M * bs_blk

    # TH: [Tk, Hk], where Tk = B*KV
    k_th = ops.Reshape()(k_raw, (B * KV, Hk))
    v_th = ops.Reshape()(v_raw, (B * KV, Hk))

    # 3c) (Optional) GQA head align IN TH so KV hidden matches Q hidden
    nh = int(self.num_heads_per_partition)
    d  = self.head_dim
    kv_heads = Hk // d
    if self.use_gqa and kv_heads != nh:
        rep = nh // max(kv_heads, 1)
        k4 = ops.Reshape()(k_th, (B * KV, kv_heads, d))       # [Tk, n_kv, d]
        v4 = ops.Reshape()(v_th, (B * KV, kv_heads, d))
        k4 = mint.repeat_interleave(k4, repeats=rep, dim=1)   # -> [Tk, nh, d]
        v4 = mint.repeat_interleave(v4, repeats=rep, dim=1)
        k_th = ops.Reshape()(k4, (B * KV, nh * d))            # [Tk, Hq]
        v_th = ops.Reshape()(v4, (B * KV, nh * d))            # [Tk, Hq]
        # Hq remains nh*d as computed above

    # 3d) Cast FA inputs to compute dtype (prevents BF16/FP16 signature errors)
    if query.dtype != fa_dtype: query = ops.Cast()(query, fa_dtype)
    if k_th.dtype  != fa_dtype: k_th  = ops.Cast()(k_th,  fa_dtype)
    if v_th.dtype  != fa_dtype: v_th  = ops.Cast()(v_th,  fa_dtype)
    if (alibi_mask is not None) and (alibi_mask.dtype != fa_dtype):
        alibi_mask = ops.Cast()(alibi_mask, fa_dtype)

    # 3e) FlashAttention (TH) over later-chunk prefill
    #     Q: [B*S_cur, H],  K/V: [B*KV, H]
    context_th = self.flash_attention(
        query, k_th, v_th,
        attn_mask,            # or None if your FA is built as causal internally
        alibi_mask,
        None, None,
        q_seq_lens,           # actual_seq_qlen (length-B vector)
        batch_valid_length    # actual_seq_kvlen (length-B vector)
    )

    # 3f) TH -> [B,S,H] for projection, then EARLY RETURN
    context_layer = ops.Reshape()(context_th, (B, S_cur, Hq))   # [B, S, H]
    attn_out = self.wo(context_layer)  # <<< change to self.o_proj if that's your proj name
    return attn_out
# ===================== END LATER-CHUNK FAST PATH =====================

# ===================== FIRST ITERATION (TH) — ORIGINAL PATH + dtype guard =====================
# Your file already calls self.flash_attention(query, key, value, ... ) here on first iteration.
# Add only the casts RIGHT BEFORE that existing call:
if self.is_first_iteration and self.use_flash_attention and (len(query.shape) == 2):
    if query.dtype != fa_dtype: query = ops.Cast()(query, fa_dtype)
    if key.dtype   != fa_dtype: key   = ops.Cast()(key,   fa_dtype)
    if value.dtype != fa_dtype: value = ops.Cast()(value, fa_dtype)
    if (alibi_mask is not None) and (alibi_mask.dtype != fa_dtype):
        alibi_mask = ops.Cast()(alibi_mask, fa_dtype)
    # (then call your existing self.flash_attention(...) exactly as the file already does)

# ===================== DECODE (S==1) OR FA DISABLED — ORIGINAL PATH =====================
# Let your original decode branch run next (paged_attn + projection/return) with no changes.
# ========================= PATCH END =========================

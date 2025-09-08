# ========================= START PATCH (inside: if self.use_past:) =========================
# (A) First: DO NOT cast x/key/value before the cache write. Keep your existing write as-is:
# key_out = self.paged_attention_mgr(
#     key, value, slot_mapping, batch_valid_length,
#     key_cache=key_cache, value_cache=value_cache
# )
# query = ops.depend(query, key_out)  # write-before-read fence

# (B) Determine batch and current chunk length S_cur from runtime shapes (x may be TH)
if len(x.shape) == 3:  # [B, S, H]
    B = int(x.shape[0]); S_cur = int(x.shape[1])
else:                  # [T, H]  (your logs show this)
    B = int(block_tables.shape[0])
    T = int(x.shape[0])
    S_cur = T if B == 1 else T // max(B, 1)

# ===================== 1) FIRST-PREFILL (TH) — FlashAttention with mask =====================
if self.is_first_iteration and self.use_flash_attention:
    # Cast ONLY at the FA boundary (keeps paged-cache kernels in their native dtype)
    q_fa = query   # TH [T, H]
    k_fa = key     # TH [T, H] in your run
    v_fa = value   # TH [T, H]
    if q_fa.dtype != self.compute_dtype: q_fa = self.cast(q_fa, self.compute_dtype)   # BF16
    if k_fa.dtype != self.compute_dtype: k_fa = self.cast(k_fa, self.compute_dtype)
    if v_fa.dtype != self.compute_dtype: v_fa = self.cast(v_fa, self.compute_dtype)

    # Length vectors must be int32 and colocated with Q to avoid heterogeneous copies
    if q_seq_lens.dtype != mstype.int32:         q_seq_lens        = self.cast(q_seq_lens,        mstype.int32)
    if batch_valid_length.dtype != mstype.int32: batch_valid_length = self.cast(batch_valid_length, mstype.int32)
    q_seq_lens         = ops.depend(q_seq_lens, q_fa)
    batch_valid_length = ops.depend(batch_valid_length, q_fa)
    k_fa = ops.depend(k_fa, q_fa)
    v_fa = ops.depend(v_fa, q_fa)

    # Normalize attn_mask: BF16/FP* -> bool, colocated with Q
    _attn_mask = attn_mask
    if _attn_mask is not None:
        zero = ops.zeros((), _attn_mask.dtype)
        _attn_mask = ops.gt(_attn_mask, zero)      # bool mask (True = keep)
        _attn_mask = ops.depend(_attn_mask, q_fa)
    else:
        _attn_mask = None
    _alibi_mask = None  # per your logs

    # FlashAttention wrapper (input_layout="TH") — exact arg order for v1.6
    context_first = self.flash_attention(
        q_fa, k_fa, v_fa,
        _attn_mask, _alibi_mask,
        None, None,
        q_seq_lens, batch_valid_length
    )
    # Continue as your original code expects (if it reshapes to [B,S,H] before projection, keep doing that)
    context_layer = context_first  # TH or already reshaped by your code later
    # (Do NOT early-return here if your original code has subsequent reshape/projection steps tied to first-iter.)
# ================== END 1) FIRST-PREFILL (TH) ==================


# ===================== 2) LATER-PREFILL (CHUNKED) — TH FlashAttention (no mask) =====================
elif self.use_flash_attention and (S_cur > 1):
    # Ensure Q is TH [B*S_cur, H]
    if len(query.shape) == 3:  # [B, S, H] -> TH
        Hq = int(query.shape[-1])
        query_th = self.reshape(query, (B * S_cur, Hq))
    else:
        query_th = query
        Hq = int(query_th.shape[-1])

    # Gather KV blocks from paged cache → usually [B*M, bs_block, Hk]
    if block_tables.dtype != mstype.int32:
        block_tables = self.cast(block_tables, mstype.int32)
    M = int(block_tables.shape[1])
    flat = self.reshape(block_tables, (B * M,))               # [B*M]
    k_raw = self.gather(key_cache,   flat, 0)
    v_raw = self.gather(value_cache, flat, 0)

    # Flatten KV to TH. Support both 3D ([B*M, bs, Hk]) and 4D ([..., N, D]) cache shapes.
    if len(k_raw.shape) == 3:
        bs_blk = int(k_raw.shape[1]); Hk = int(k_raw.shape[2]); KV = M * bs_blk
        k_th = self.reshape(k_raw, (B * KV, Hk))               # [Tk, Hk]
        v_th = self.reshape(v_raw, (B * KV, Hk))               # [Tk, Hk]
    else:
        N = int(k_raw.shape[-2]); D = int(k_raw.shape[-1]); Hk = N * D
        Tk = 1
        for d_ in k_raw.shape[:-2]:
            Tk *= int(d_)
        k_th = self.reshape(k_raw, (Tk, Hk))
        v_th = self.reshape(v_raw, (Tk, Hk))
        KV = Tk // max(B, 1)

    # Optional GQA align in TH (tile KV heads n_kv -> n_q so hidden matches Q: 512 -> 3584)
    nh = int(self.num_heads_per_partition)    # e.g., 28
    d  = self.head_dim                         # e.g., 128
    kv_heads = Hk // d                         # e.g., 4
    if getattr(self, "use_gqa", False) and kv_heads != nh:
        rep = nh // max(kv_heads, 1)          # e.g., 7
        k4 = self.reshape(k_th, (B * KV, kv_heads, d))         # [Tk, n_kv, d]
        v4 = self.reshape(v_th, (B * KV, kv_heads, d))
        # Use device tile (not mint.repeat_interleave) to avoid host ops
        k4 = ops.tile(k4, (1, rep, 1))                             # -> [Tk, nh, d]
        v4 = ops.tile(v4, (1, rep, 1))
        k_th = self.reshape(k4, (B * KV, nh * d))                  # -> [Tk, Hq]
        v_th = self.reshape(v4, (B * KV, nh * d))                  # -> [Tk, Hq]
        Hq = nh * d

    # Cast ONLY at the FA boundary (BF16) and colocate control tensors with Q
    q_fa = query_th
    k_fa = k_th
    v_fa = v_th
    if q_fa.dtype != self.compute_dtype: q_fa = self.cast(q_fa, self.compute_dtype)
    if k_fa.dtype  != self.compute_dtype: k_fa = self.cast(k_fa,  self.compute_dtype)
    if v_fa.dtype  != self.compute_dtype: v_fa = self.cast(v_fa,  self.compute_dtype)

    if q_seq_lens.dtype != mstype.int32:         q_seq_lens        = self.cast(q_seq_lens,        mstype.int32)
    if batch_valid_length.dtype != mstype.int32: batch_valid_length = self.cast(batch_valid_length, mstype.int32)
    q_seq_lens         = ops.depend(q_seq_lens, q_fa)
    batch_valid_length = ops.depend(batch_valid_length, q_fa)
    k_fa = ops.depend(k_fa, q_fa)
    v_fa = ops.depend(v_fa, q_fa)

    # No masks here: rely on causal FA + length vectors (safer for hetero-copy)
    _attn_mask  = None
    _alibi_mask = None

    # FlashAttention (TH) — wrapper arg order
    context_th = self.flash_attention(
        q_fa, k_fa, v_fa,
        _attn_mask, _alibi_mask,
        None, None,
        q_seq_lens, batch_valid_length
    )

    # TH → [B, S, H] for projection; EARLY RETURN
    context_layer = self.reshape(context_th, (B, S_cur, Hq))
    out = self.o_proj(context_layer)  # <- change to self.wo if that's your proj name
    return out
# ================== END 2) LATER-PREFILL (TH) ==================

# (Else): fall through to your original decode path (S_cur == 1 → paged attention) and any other branches
# ========================== END PATCH ==========================

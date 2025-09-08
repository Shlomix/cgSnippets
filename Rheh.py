# ========================= PATCH: first-iteration FA call (TH) =========================
# Assumptions in your run:
# - Layout is TH: query/key/value are [T, H]
# - attn_mask is NOT None (shape e.g. 128x128) and arrives as BF16
# - alibi_mask is None
# - KV has already been written to cache earlier (do NOT add any casts before that write)

if self.use_flash_attention:
    # 1) Cast ONLY at the FA boundary (keeps cache kernels happy)
    q_fa = query
    k_fa = key
    v_fa = value
    if q_fa.dtype != self.compute_dtype:
        q_fa = self.cast(q_fa, self.compute_dtype)      # BF16
    if k_fa.dtype != self.compute_dtype:
        k_fa = self.cast(k_fa, self.compute_dtype)
    if v_fa.dtype != self.compute_dtype:
        v_fa = self.cast(v_fa, self.compute_dtype)

    # 2) Length vectors must be int32 and colocated with Q to avoid heterogeneous copies
    if q_seq_lens.dtype != mstype.int32:
        q_seq_lens = self.cast(q_seq_lens, mstype.int32)
    if batch_valid_length.dtype != mstype.int32:
        batch_valid_length = self.cast(batch_valid_length, mstype.int32)
    q_seq_lens         = ops.depend(q_seq_lens, q_fa)
    batch_valid_length = ops.depend(batch_valid_length, q_fa)
    k_fa = ops.depend(k_fa, q_fa)
    v_fa = ops.depend(v_fa, q_fa)

    # 3) Normalize attn_mask for FA: convert BF16 → bool and colocate with Q
    _attn_mask = attn_mask
    if _attn_mask is not None:
        zero = ops.zeros((), _attn_mask.dtype)
        _attn_mask = ops.gt(_attn_mask, zero)   # bool mask (True = keep)
        _attn_mask = ops.depend(_attn_mask, q_fa)
    else:
        _attn_mask = None

    _alibi_mask = None  # per your logs

    # 4) Call FlashAttention (wrapper arg order for MindFormers v1.6; input_layout="TH")
    context_layer = self.flash_attention(
        q_fa,               # query  [Tq, H]
        k_fa,               # key    [Tk, H]
        v_fa,               # value  [Tk, H]
        _attn_mask,         # attn_mask (bool or None)
        _alibi_mask,        # alibi_mask or None
        None,               # prefix
        None,               # padding_mask
        q_seq_lens,         # actual_seq_qlen   (shape [B], int32)
        batch_valid_length  # actual_seq_kvlen  (shape [B], int32)
    )

    # Continue exactly as your original code (reshape to [B,S,H] if needed, then projection)
else:
    # Original non-FA path (unchanged)
    context_layer = self.core_attention(query, key, value, attn_mask)
# ====================== END PATCH: first-iteration FA call (TH) ======================

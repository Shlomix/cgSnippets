# ===================== INSERT INSIDE ParallelAttention.construct(...), under: if self.use_past: =====================

# (0) Stabilize graph input signature once (avoid BF16/FP16 signature flips)
if x.dtype != self.compute_dtype:
    x = self.cast(x, self.compute_dtype)

# ... your existing Q/K/V projection + split happen above ...

# (1) ALWAYS write the current chunk's K/V to paged cache (unchanged)
key_out = self.paged_attention_mgr(
    key, value, slot_mapping, batch_valid_length,
    key_cache=key_cache, value_cache=value_cache
)
query = ops.depend(query, key_out)  # fence: write-before-read

# (2) Robust batch & current-chunk length from runtime shapes
if len(x.shape) == 3:        # [B, S, H]
    B = int(x.shape[0]); S_cur = int(x.shape[1])
else:                        # [T, H]  (your case)
    B = int(block_tables.shape[0])    # e.g., 1
    S_cur = int(x.shape[0]) if B == 1 else int(x.shape[0]) // max(B, 1)

# ===================== LATER-PREFILL FAST PATH (TH) =====================
# Intercept only AFTER the first iteration, and only when this is a true prefill chunk (S_cur > 1)
if self.use_flash_attention and (not self.is_first_iteration) and (S_cur > 1):
    # 2a) Ensure Q is TH: [B,S,H] -> [Tq,H] if needed
    if len(query.shape) == 3:
        Hq = int(query.shape[-1])
        query = self.reshape(query, (B * S_cur, Hq))        # -> [Tq, H]
    else:
        Hq = int(query.shape[-1])                           # already [Tq, H]

    # 2b) Gather ALL KV blocks from cache and FLATTEN to TH directly
    if block_tables.dtype != mstype.int32:
        block_tables = self.cast(block_tables, mstype.int32)
    M = int(block_tables.shape[1])                          # e.g., 500
    flat = self.reshape(block_tables, (B * M,))             # [B*M]

    # cache layout usually: [B*M, bs_block, Hk]  (but we also tolerate 4D)
    k_raw = self.gather(key_cache,   flat, 0)
    v_raw = self.gather(value_cache, flat, 0)

    if len(k_raw.shape) == 3:
        bs_blk = int(k_raw.shape[1]); Hk = int(k_raw.shape[2])
        KV = M * bs_blk
        k_th = self.reshape(k_raw, (B * KV, Hk))            # -> [Tk, Hk]
        v_th = self.reshape(v_raw, (B * KV, Hk))            # -> [Tk, Hk]
    else:
        # generic 4D fallback: [..., N, D] -> TH by collapsing heads
        N  = int(k_raw.shape[-2]); D = int(k_raw.shape[-1])
        Hk = N * D
        Tk = 1
        for dim_i in k_raw.shape[:-2]:
            Tk *= int(dim_i)
        k_th = self.reshape(k_raw, (Tk, Hk))
        v_th = self.reshape(v_raw, (Tk, Hk))
        KV = Tk // max(B, 1)

    # 2c) GQA head align in TH (tile KV heads 4 -> 28 so hidden matches Q: 512 -> 3584)
    nh = int(self.num_heads_per_partition)   # 28
    d  = self.head_dim                       # 128
    kv_heads = Hk // d                       # 4
    if self.use_gqa and kv_heads != nh:
        rep = nh // max(kv_heads, 1)         # 28/4 = 7
        k4 = self.reshape(k_th, (B * KV, kv_heads, d))  # [Tk, 4, 128]
        v4 = self.reshape(v_th, (B * KV, kv_heads, d))
        k4 = mint.repeat_interleave(k4, repeats=rep, dim=1)  # -> [Tk, 28, 128]
        v4 = mint.repeat_interleave(v4, repeats=rep, dim=1)
        k_th = self.reshape(k4, (B * KV, nh * d))       # -> [Tk, 3584]
        v_th = self.reshape(v4, (B * KV, nh * d))       # -> [Tk, 3584]
        Hq = nh * d                                     # keep hidden consistent (=3584)

    # 2d) Sanitize FA inputs: keep mask None; control lengths int32 & colocated; cast Q/K/V to compute dtype (BF16)
    _mask  = None
    _alibi = None  # set to cast(alibi_mask, self.compute_dtype) if you truly need it

    if query.dtype != self.compute_dtype: query = self.cast(query, self.compute_dtype)
    if k_th.dtype  != self.compute_dtype: k_th  = self.cast(k_th,  self.compute_dtype)
    if v_th.dtype  != self.compute_dtype: v_th  = self.cast(v_th,  self.compute_dtype)

    if q_seq_lens.dtype != mstype.int32:
        q_seq_lens = self.cast(q_seq_lens, mstype.int32)
    if batch_valid_length.dtype != mstype.int32:
        batch_valid_length = self.cast(batch_valid_length, mstype.int32)
    q_seq_lens         = ops.depend(q_seq_lens, query)          # colocate to avoid heterogeneous copy
    batch_valid_length = ops.depend(batch_valid_length, query)

    # 2e) FlashAttention (TH) over later prefill
    #     Q: [B*S_cur, Hq]  (e.g., [4096, 3584]),
    #     K/V: [B*KV,  Hq]  (KV derived from blocks; Hq after GQA = 3584)
    context_th = self.flash_attention(
        query, k_th, v_th,
        _mask, _alibi,
        None, None,
        q_seq_lens,            # shape [B] (=1)
        batch_valid_length     # shape [B] (=1)
    )

    # 2f) TH -> [B,S,H] for projection, then EARLY RETURN
    context_layer = self.reshape(context_th, (B, S_cur, Hq))    # -> [1, 4096, 3584]
    attn_out = self.o_proj(context_layer)  # <-- if your proj is named `wo`, change to `self.wo`
    return attn_out

# (else) fall through to your ORIGINAL first-iteration (TH) path and decode path unchanged.
# ====================================================================================================================

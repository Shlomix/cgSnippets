# ===== inside ParallelAttention.construct(...), under: if self.use_past: =====
from mindspore import dtype as mstype

# 0) stabilize input signature once; no edits in first-iter code paths needed
if x.dtype != self.compute_dtype:
    x = self.cast(x, self.compute_dtype)

# ... your existing code that builds query/key/value ...

# 1) ALWAYS write the current chunk's KV to the paged cache (unchanged)
key_out = self.paged_attention_mgr(
    key, value, slot_mapping, batch_valid_length,
    key_cache=key_cache, value_cache=value_cache
)
query = ops.depend(query, key_out)  # fence: write-before-read

# 2) robust chunk length (works even when x and query are TH)
if len(x.shape) == 3:           # x: [B, S, H]
    B = int(x.shape[0])
    S_cur = int(x.shape[1])
else:                           # x: [T, H]
    B = int(block_tables.shape[0])
    S_cur = int(x.shape[0]) if B == 1 else int(x.shape[0]) // max(B, 1)

# ===================== LATER-CHUNK CHUNK-PREFILL FAST PATH (TH) =====================
# Only intercept when NOT first iteration and we truly have a prefill chunk (S_cur > 1)
if self.use_flash_attention and (not self.is_first_iteration) and (S_cur > 1):
    # Ensure Q is TH: [B,S,H] -> [Tq,H] if needed
    if len(query.shape) == 3:
        Hq = int(query.shape[-1])
        query = self.reshape(query, (B * S_cur, Hq))      # [Tq, H]
    else:
        Hq = int(query.shape[-1])                         # already [Tq, H]

    # Gather ALL KV blocks from cache and FLATTEN directly to TH
    if block_tables.dtype != mstype.int32:
        block_tables = self.cast(block_tables, mstype.int32)
    M = int(block_tables.shape[1])
    flat = self.reshape(block_tables, (B * M,))           # [B*M]

    # caches layout: [B*M, bs_block, Hk]
    k_raw = self.gather(key_cache,   flat, 0)             # [B*M, bs, Hk]
    v_raw = self.gather(value_cache, flat, 0)
    bs_blk = int(k_raw.shape[1]); Hk = int(k_raw.shape[-1])
    KV = M * bs_blk

    # TH for K/V: [Tk, Hk] where Tk = B*KV
    k_th = self.reshape(k_raw, (B * KV, Hk))
    v_th = self.reshape(v_raw, (B * KV, Hk))

    # Optional GQA head align IN TH so KV hidden matches Q hidden (Hq = nh*d)
    nh = int(self.num_heads_per_partition)
    d  = self.head_dim
    kv_heads = Hk // d
    if self.use_gqa and kv_heads != nh:
        rep = nh // max(kv_heads, 1)
        k4 = self.reshape(k_th, (B * KV, kv_heads, d))    # [Tk, n_kv, d]
        v4 = self.reshape(v_th, (B * KV, kv_heads, d))
        k4 = mint.repeat_interleave(k4, repeats=rep, dim=1)  # -> [Tk, nh, d]
        v4 = mint.repeat_interleave(v4, repeats=rep, dim=1)
        k_th = self.reshape(k4, (B * KV, nh * d))         # [Tk, Hq]
        v_th = self.reshape(v4, (B * KV, nh * d))         # [Tk, Hq]
        Hq = nh * d                                       # keep Hq consistent

    # Cast FA inputs to compute dtype (prevents BF16/FP16 signature errors)
    if query.dtype != self.compute_dtype: query = self.cast(query, self.compute_dtype)
    if k_th.dtype  != self.compute_dtype: k_th  = self.cast(k_th,  self.compute_dtype)
    if v_th.dtype  != self.compute_dtype: v_th  = self.cast(v_th,  self.compute_dtype)
    if (alibi_mask is not None) and (alibi_mask.dtype != self.compute_dtype):
        alibi_mask = self.cast(alibi_mask, self.compute_dtype)

    # FlashAttention (TH) for chunk prefill (later chunks)
    # Q: [B*S_cur, Hq],  K/V: [B*KV, Hq]
    context_th = self.flash_attention(
        query, k_th, v_th,
        attn_mask,            # or None if your FA kernel encodes causality
        alibi_mask,
        None, None,
        q_seq_lens,           # actual_seq_qlen  (shape [B])
        batch_valid_length    # actual_seq_kvlen (shape [B])
    )

    # TH -> [B,S,H] for projection, then EARLY RETURN
    context_layer = self.reshape(context_th, (B, S_cur, Hq))   # [B, S, H]
    attn_out = self.wo(context_layer)  # <-- if your proj is `o_proj`, change to `self.o_proj`
    return attn_out
# ===================== END LATER-CHUNK FAST PATH =====================

# else: fall through to your UNCHANGED original first-iteration (TH) path and decode path
# (no edits below this line)

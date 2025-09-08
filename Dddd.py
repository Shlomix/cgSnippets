# ============================== FULL PATCH (copy–paste) ==============================
# File: research/qwen2_5/infer/transformer.py  (ParallelAttention class)
# Adds two TH-FlashAttention cells (first-prefill with mask, later-prefill without mask),
# and replaces the FA call sites inside `construct(...)` under `if self.use_past:`.
# IMPORTANT:
#   - Do NOT cast `x`, `key`, or `value` before the cache write.
#   - Projection layer name below assumes `self.o_proj`. If yours is `self.wo`, change that one line.

# -------- 1) ADD THESE IMPORTS NEAR THE TOP OF THE FILE (if not present) --------
from mindspore import ops
from mindspore import dtype as mstype
from mindformers.modules.flash_attention import FlashAttention
# ---------------------------------------------------------------------------------


# -------- 2) IN ParallelAttention.__init__(...), ADD THESE LINES ------------------
# (Place them after heads/scale are known, and after self.use_past / self.use_flash_attention are set.)
if getattr(self, "use_past", False) and getattr(self, "use_flash_attention", False):
    # TH FA for the very first prefill (we WILL pass an attention mask + lengths)
    self.fa_first_th = FlashAttention(
        head_num=self.num_heads_per_partition,
        keep_prob=1.0,
        scale_value=1.0 / self.norm_factor,
        input_layout="TH",
        sparse_mode=0,
        use_attention_mask=True,
        use_actual_seqlen=True
    )
    # TH FA for later prefill chunks (NO explicit mask; rely on actual seq lengths)
    self.fa_later_th = FlashAttention(
        head_num=self.num_heads_per_partition,
        keep_prob=1.0,
        scale_value=1.0 / self.norm_factor,
        input_layout="TH",
        sparse_mode=0,
        use_attention_mask=False,
        use_actual_seqlen=True
    )
# ---------------------------------------------------------------------------------


# -------- 3) IN ParallelAttention.construct(...), INSIDE: `if self.use_past:` ----
# Keep your existing Q/K/V projection + split ABOVE this point.

# (A) ALWAYS write current K/V to the paged cache FIRST — UNCHANGED
key_out = self.paged_attention_mgr(
    key, value, slot_mapping, batch_valid_length,
    key_cache=key_cache, value_cache=value_cache
)
# Fence to ensure the write completes before any read path:
query = ops.depend(query, key_out)

# (B) Derive batch and current chunk length S_cur from runtime shapes
if len(x.shape) == 3:  # [B, S, H]
    B = int(x.shape[0]); S_cur = int(x.shape[1])
else:                  # [T, H]  (your run)
    B = int(block_tables.shape[0])
    T = int(x.shape[0])
    S_cur = T if B == 1 else T // max(B, 1)

# ==================== 1) FIRST-PREFILL (TH) — FlashAttention WITH mask ====================
if self.is_first_iteration and self.use_flash_attention:
    # Cast ONLY at the FA boundary (do NOT cast x/key/value before cache write)
    q_fa = query   # TH [T, H]
    k_fa = key     # TH [T, H]
    v_fa = value   # TH [T, H]
    if q_fa.dtype != self.compute_dtype: q_fa = self.cast(q_fa, self.compute_dtype)   # e.g., BF16
    if k_fa.dtype != self.compute_dtype: k_fa = self.cast(k_fa, self.compute_dtype)
    if v_fa.dtype != self.compute_dtype: v_fa = self.cast(v_fa, self.compute_dtype)

    # Length vectors must be int32 and colocated with Q to avoid heterogeneous copies
    if q_seq_lens.dtype != mstype.int32:         q_seq_lens        = self.cast(q_seq_lens,        mstype.int32)
    if batch_valid_length.dtype != mstype.int32: batch_valid_length = self.cast(batch_valid_length, mstype.int32)
    q_seq_lens         = ops.depend(q_seq_lens, q_fa)
    batch_valid_length = ops.depend(batch_valid_length, q_fa)
    k_fa = ops.depend(k_fa, q_fa)
    v_fa = ops.depend(v_fa, q_fa)

    # Normalize attn_mask: convert BF16/FP* → bool and colocate with Q
    _attn_mask = attn_mask
    if _attn_mask is not None:
        zero = ops.zeros((), _attn_mask.dtype)
        _attn_mask = ops.gt(_attn_mask, zero)   # bool: True = keep
        _attn_mask = ops.depend(_attn_mask, q_fa)
    else:
        _attn_mask = None

    # alibi not used in your run
    context_first = self.fa_first_th(
        q_fa, k_fa, v_fa,
        _attn_mask,   # attn_mask (bool or None)
        None,         # alibi_mask
        None, None,   # prefix, padding_mask
        q_seq_lens, batch_valid_length
    )
    # If your original code expects TH here and later reshapes, you can pass it on;
    # otherwise, project and return (choose ONE style; here we project & return to be self-contained):
    # TH -> [B, S, H] and project:
    Hq_first = int(context_first.shape[-1])
    context_layer = ops.Reshape()(context_first, (B, S_cur, Hq_first))
    out_first = self.o_proj(context_layer)  # CHANGE to self.wo(...) if that's your proj
    return out_first

# ==================== 2) LATER-PREFILL (TH) — FlashAttention WITHOUT mask ====================
elif self.use_flash_attention and (S_cur > 1):
    # Ensure Q is TH [B*S_cur, H]
    if len(query.shape) == 3:  # [B, S, H] -> TH
        Hq = int(query.shape[-1])
        query_th = ops.Reshape()(query, (B * S_cur, Hq))
    else:
        query_th = query
        Hq = int(query_th.shape[-1])

    # Gather KV blocks from paged cache → flatten to TH
    if block_tables.dtype != mstype.int32:
        block_tables = self.cast(block_tables, mstype.int32)
    M = int(block_tables.shape[1])
    flat = ops.Reshape()(block_tables, (B * M,))                 # [B*M]
    k_raw = self.gather(key_cache,   flat, 0)                    # usually [B*M, bs, Hk]
    v_raw = self.gather(value_cache, flat, 0)

    if len(k_raw.shape) == 3:
        bs_blk = int(k_raw.shape[1]); Hk = int(k_raw.shape[2]); KV = M * bs_blk
        k_th = ops.Reshape()(k_raw, (B * KV, Hk))                # [Tk, Hk]
        v_th = ops.Reshape()(v_raw, (B * KV, Hk))                # [Tk, Hk]
    else:
        # Generic 4D fallback: [..., N, D] -> collapse to TH
        N = int(k_raw.shape[-2]); D = int(k_raw.shape[-1]); Hk = N * D
        Tk = 1
        for d_ in k_raw.shape[:-2]:
            Tk *= int(d_)
        k_th = ops.Reshape()(k_raw, (Tk, Hk))
        v_th = ops.Reshape()(v_raw, (Tk, Hk))
        KV = Tk // max(B, 1)

    # Optional GQA align in TH (tile KV heads: n_kv -> n_q so hidden matches Q)
    nh = int(self.num_heads_per_partition)   # e.g., 28
    d  = self.head_dim                        # e.g., 128
    kv_heads = Hk // d                        # e.g., 4
    if getattr(self, "use_gqa", False) and kv_heads != nh:
        rep = nh // max(kv_heads, 1)          # e.g., 7
        k4 = ops.Reshape()(k_th, (B * KV, kv_heads, d))         # [Tk, n_kv, d]
        v4 = ops.Reshape()(v_th, (B * KV, kv_heads, d))
        # Use device tile (not mint.repeat_interleave) to avoid host ops
        k4 = ops.tile(k4, (1, rep, 1))                               # -> [Tk, nh, d]
        v4 = ops.tile(v4, (1, rep, 1))
        k_th = ops.Reshape()(k4, (B * KV, nh * d))                   # -> [Tk, Hq]
        v_th = ops.Reshape()(v4, (B * KV, nh * d))                   # -> [Tk, Hq]
        Hq = nh * d

    # Cast ONLY at FA boundary; lengths int32 + colocated
    q_fa = query_th; k_fa = k_th; v_fa = v_th
    if q_fa.dtype != self.compute_dtype: q_fa = self.cast(q_fa, self.compute_dtype)
    if k_fa.dtype  != self.compute_dtype: k_fa = self.cast(k_fa,  self.compute_dtype)
    if v_fa.dtype  != self.compute_dtype: v_fa = self.cast(v_fa,  self.compute_dtype)

    if q_seq_lens.dtype != mstype.int32:         q_seq_lens        = self.cast(q_seq_lens,        mstype.int32)
    if batch_valid_length.dtype != mstype.int32: batch_valid_length = self.cast(batch_valid_length, mstype.int32)
    q_seq_lens         = ops.depend(q_seq_lens, q_fa)
    batch_valid_length = ops.depend(batch_valid_length, q_fa)
    k_fa = ops.depend(k_fa, q_fa)
    v_fa = ops.depend(v_fa, q_fa)

    # NO masks here; rely on lengths + causal FA
    context_th = self.fa_later_th(
        q_fa, k_fa, v_fa,
        None, None,   # attn_mask, alibi_mask
        None, None,   # prefix, padding_mask
        q_seq_lens, batch_valid_length
    )

    # TH → [B, S, H] for projection; EARLY RETURN
    context_layer = ops.Reshape()(context_th, (B, S_cur, Hq))
    out = self.o_proj(context_layer)  # CHANGE to self.wo(...) if that's your proj name
    return out

# (Else): fall through to your original decode path (S_cur == 1 → paged attention) and any other branches
# ============================== END FULL PATCH =================================

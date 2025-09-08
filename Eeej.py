# ======================== LATER-PREFILL-ONLY PATCH (copy–paste) ========================
# File: research/qwen2_5/infer/transformer.py  (ParallelAttention class)

# --- 1) Add (or ensure) these imports near the top of the file ---
from mindspore import ops
from mindspore import dtype as mstype
from mindformers.modules.flash_attention import FlashAttention
# ----------------------------------------------------------------


# --- 2) In ParallelAttention.__init__(...), ADD this small snippet ---
#     (Place it after heads/scale are known, and after self.use_past / self.use_flash_attention are set.)
if getattr(self, "use_past", False) and getattr(self, "use_flash_attention", False):
    # TH FlashAttention for LATER prefill (no explicit mask; rely on actual seq lengths)
    self.fa_later_th = FlashAttention(
        head_num=self.num_heads_per_partition,
        keep_prob=1.0,
        scale_value=1.0 / self.norm_factor,
        input_layout="TH",
        sparse_mode=0,
        use_attention_mask=False,   # we won't pass a mask on later chunks
        use_actual_seqlen=True      # we WILL pass q/kv lengths
    )
# ------------------------------------------------------------------


# --- 3) In ParallelAttention.construct(...), inside: `if self.use_past:` ---
# Keep your existing Q/K/V projection + split ABOVE this point.

# (A) ALWAYS write current K/V to paged cache FIRST — DO NOT CHANGE THIS
key_out = self.paged_attention_mgr(
    key, value, slot_mapping, batch_valid_length,
    key_cache=key_cache, value_cache=value_cache
)
# Fence to ensure the write completes before any read path:
query = ops.depend(query, key_out)

# (B) Derive batch and current chunk length S_cur from runtime shapes
if len(x.shape) == 3:  # [B, S, H]
    B = int(x.shape[0]); S_cur = int(x.shape[1])
else:                  # [T, H]  (your logs)
    B = int(block_tables.shape[0])
    T = int(x.shape[0])
    S_cur = T if B == 1 else T // max(B, 1)

# ===================== LATER-PREFILL FAST PATH (TH FlashAttention) =====================
# Only intercept later prefill (NOT first iteration) and ONLY when S_cur > 1 (not decode).
if (not self.is_first_iteration) and getattr(self, "use_flash_attention", False) and (S_cur > 1):
    # Ensure Q is TH: [B,S,H] -> [B*S, H] if needed
    if len(query.shape) == 3:
        Hq = int(query.shape[-1])
        query_th = ops.Reshape()(query, (B * S_cur, Hq))
    else:
        query_th = query
        Hq = int(query_th.shape[-1])

    # Gather KV from paged cache → flatten to TH
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

    # Optional GQA align in TH (tile KV heads n_kv -> n_q so hidden matches Q)
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

    # Cast ONLY at FA boundary; lengths int32 + colocated (prevents hetero-copy)
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

    # NO masks here; rely on causal FA + actual_seq_* lengths
    context_th = self.fa_later_th(
        q_fa, k_fa, v_fa,
        None, None,     # attn_mask, alibi_mask
        None, None,     # prefix, padding_mask
        q_seq_lens, batch_valid_length
    )

    # TH → [B, S, H] for projection; EARLY RETURN
    context_layer = ops.Reshape()(context_th, (B, S_cur, Hq))
    out = self.o_proj(context_layer)  # <-- if your proj is `wo`, change to `self.wo`
    return out

# ELSE: fall through to your original branches:
# - first prefill (your existing FA code stays intact)
# - decode (S_cur == 1) -> original paged attention path
# ===================== END LATER-PREFILL-ONLY PATCH =====================

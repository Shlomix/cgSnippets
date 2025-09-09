# FILE: mindformers/modules/infer_attention.py
# GOAL: Use FlashAttention only for 2D+ prefill chunks from the second chunk onward.
#       Keep first prefill and all decode on paged attention. Always update paged KV cache.

# 1) At top-level imports, add (near other modules imports):
from mindformers.modules.flash_attention import FlashAttention  # reuse built-in FA
import os

# 2) Inside class InferAttention.__init__(...), after existing flags (e.g., use_flash_attention), add:
self._use_fa_prefill_2d = bool(os.getenv("MF_FA_PREFILL_2D", "0") == "1")

# Reuse existing FA instance if already present; otherwise create exactly one with canonical args.
self.flash_attention = getattr(self, "flash_attention", None)
if self.use_flash_attention and self.flash_attention is None:
    # Match MindFormers FA API so only one kernel specialization is compiled.
    # Layout is BNSD; sparse_mode=2 is causal (no explicit mask required).
    self.flash_attention = FlashAttention(
        head_num=self.num_heads,           # or self.num_attention_heads_per_partition if that’s what the file uses
        keep_prob=1.0,
        input_layout="BNSD",
        sparse_mode=2,                     # causal
        use_attention_mask=False,
        use_actual_seqlen=True
    )
    # If this module supports sharding, keep same parallel config as the rest of the file
    if hasattr(self.flash_attention, "shard") and hasattr(self, "parallel_config"):
        self.flash_attention.shard(self.parallel_config)

# 3) Inside InferAttention.construct(...), after you have q, k, v in (B, N, S, D) and after RoPE,
#    insert the following gate BEFORE the normal paged-attention path:

# ---------- FlashAttention gate for 2D+ prefill (skip very first chunk) ----------
# Preconditions for FA:
#   - Global FlashAttention enabled AND our 2D prefill toggle set
#   - q_seq_lens is provided (that’s how 2D+ prefill is signaled)
#   - batch_valid_length > 0 (means we are NOT on the very first prefill chunk)
fa_ready = (
    getattr(self, "use_flash_attention", False)
    and getattr(self, "flash_attention", None) is not None
    and self._use_fa_prefill_2d
    and (q_seq_lens is not None)
)

# Determine if we’re on chunk > 0: batch_valid_length holds prefix length *before* this chunk.
# Works when batch_valid_length is a Tensor or Python list.
def _get_max_prefix_len(x):
    try:
        from mindspore import ops as P
        return int(P.ReduceMax(keep_dims=False)(x).asnumpy())
    except Exception:
        try:
            return int(max(x))
        except Exception:
            return 0

not_first_prefill_chunk = _get_max_prefix_len(batch_valid_length) > 0

if fa_ready and not_first_prefill_chunk:
    # (a) Always update KV paged cache for this chunk so decode stays on paged attention.
    #     This calls PagedAttentionMgr.construct(key, value, slot_mapping) under the hood.
    if slot_mapping is not None:
        _ = self.paged_attention_mgr(k, v, slot_mapping)  # caches into internal key_cache/value_cache
        # See PagedAttentionMgr.construct signature in MindFormers. :contentReference[oaicite:2]{index=2}

    # (b) Handle GQA: if K/V heads are fewer than Q heads, repeat K/V along head axis to match Q.
    #     (InferAttention already has n_rep / use_gqa in most models; adjust names if different.)
    if getattr(self, "n_rep", 1) > 1:
        from mindspore import ops as P
        k = P.Tile()(k, (1, self.n_rep, 1, 1))
        v = P.Tile()(v, (1, self.n_rep, 1, 1))

    # (c) Compute per-chunk lengths for FA:
    #     actual_seq_qlen = current chunk sizes
    #     actual_seq_kvlen = prefix (before this chunk) + current chunk sizes
    actual_qlen = q_seq_lens
    actual_kvlen = batch_valid_length + q_seq_lens

    # (d) Run FlashAttention in causal sparse mode; no attn_mask needed.
    #     API: FlashAttention(query, key, value, attn_mask=None, actual_seq_qlen=..., actual_seq_kvlen=...)
    fa_out = self.flash_attention(
        query=q, key=k, value=v,
        attn_mask=None,
        actual_seq_qlen=actual_qlen,
        actual_seq_kvlen=actual_kvlen
    )  # shape: (B, N, Sq, Dh)

    # (e) Merge heads to (B, Sq, H) to match the rest of the pipeline:
    B_, N_, Sq_, Dh_ = fa_out.shape
    context_layer = fa_out.transpose(0, 2, 1, 3).reshape(B_, Sq_, N_ * Dh_)

    # Return early so the regular paged-attn path is skipped for these chunks.
    return context_layer
# ---------- end FA 2D+ prefill gate ----------

# 4) Leave your existing paged-attention path UNCHANGED below.
#    This continues to handle the very first prefill chunk (prefix==0) and all decode steps:
#    - KV update:   self.paged_attention_mgr(k, v, slot_mapping)
#    - Compute attn: self.paged_attention_mgr.paged_attn(q, batch_valid_length, block_tables, attn_mask=..., q_seq_lens=...)
#    See method names/signatures in PagedAttentionMgr. :contentReference[oaicite:3]{index=3}

# 5) Enable with environment variable (no schema changes required):
#    export MF_FA_PREFILL_2D=1

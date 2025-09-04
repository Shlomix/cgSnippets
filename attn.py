# =================== Option B: external caches (minimal patch) ===================
# Add (if not present) near other imports
from mindspore import ops

# ---- in __init__ ---------------------------------------------------------------
# Keep your existing manager construction. DO NOT force npu_mem_size>0 here.
# The manager stays in external-cache mode (e.g., npu_mem_size == -1 in your config).
# If you created fa_prefill earlier, keep it; else you can keep using your existing FA instance.

# ---- inside construct(...), do these three very small changes ------------------

# 0) WRITE: always reshape+cache this chunk, but PASS the external caches through.
#    (This is the only line you need to ensure the upstream-provided caches get populated.)
cache_write_out = self.paged_attention_mgr(
    key, value, slot_mapping, batch_valid_length,
    key_cache=key_cache,          # << pass-through external key cache
    value_cache=value_cache       # << pass-through external value cache
)
# Keep the ordering barrier so compute happens after write:
query = ops.depend(query, cache_write_out)

# 1) (Optional) decide prefill based on q_seq_lens (robust; no branching on tensor needed)
#    We'll still gate the FA fast-path with simple Python guards below.

# 2) LATER-PREFILL FA fast-path (keep it tiny):
#    - read KV from the *external caches* we just wrote into
#    - if heads line up, run FA; otherwise fall through to your original code
if getattr(self, "use_flash_attention", False) and (q_seq_lens is not None):
    kc = key_cache      # << read from the external cache passed in
    vc = value_cache
    if (kc is not None) and (vc is not None):
        # Gather a contiguous view from cache using the block table for this batch
        B  = int(block_tables.shape[0])
        M  = int(block_tables.shape[1])
        bs = int(kc.shape[1])               # cache layout: [num_blocks, block_size, Hk]
        Hk = int(kc.shape[-1])

        flat   = ops.reshape(block_tables, (B * M,))
        k_full = ops.reshape(ops.gather(kc, flat, 0), (B, M * bs, Hk))  # (B, KV_CAP, Hk)
        v_full = ops.reshape(ops.gather(vc, flat, 0), (B, M * bs, Hk))
        KV     = M * bs

        # Minimal legality check for BNSD reshape; if it fails, fall through to original code
        nh = int(self.num_heads_per_partition)
        Hq = int(getattr(query, "shape", (0, 0, 0))[-1] or 0)
        if (nh > 0) and (Hq % nh == 0) and (Hq == Hk):
            # Convert Q/K/V from BSH -> BNSD and call your prefill FA
            S_cur = int(getattr(query, "shape", (0, 1, 0))[1] or 1)  # if TH was ever used, S_cur is first dim; here we use current shape
            D     = Hq // nh

            q_bnsd = ops.transpose(ops.reshape(query,  (B, S_cur, nh, D)), (0, 2, 1, 3))
            k_bnsd = ops.transpose(ops.reshape(k_full, (B, KV,   nh, D)), (0, 2, 1, 3))
            v_bnsd = ops.transpose(ops.reshape(v_full, (B, KV,   nh, D)), (0, 2, 1, 3))

            # Same FA call signature you already use for first prefill:
            context_layer = self.fa_prefill(
                q_bnsd, k_bnsd, v_bnsd,
                attn_mask, alibi_mask, None, None,
                q_seq_lens, batch_valid_length
            )
            return context_layer

# 3) If we didn’t return above, execute your ORIGINAL branches exactly as-is:
#    - first-iteration path (unchanged)
#    - original "not first" path (paged/core), but MAKE SURE you pass external caches:
#      self.paged_attention_mgr.paged_attn(..., key_cache=key_cache, value_cache=value_cache)
# ================================================================================


# BEFORE:
# context_layer = self.paged_attention_mgr.paged_attn(query, batch_valid_length, block_tables,
#                                                     attn_mask=attn_mask, q_seq_lens=q_seq_lens)

# AFTER (Option B needs this):
context_layer = self.paged_attention_mgr.paged_attn(
    query, batch_valid_length, block_tables,
    attn_mask=attn_mask, q_seq_lens=q_seq_lens,
    key_cache=key_cache,          # << pass-through
    value_cache=value_cache       # << pass-through
)


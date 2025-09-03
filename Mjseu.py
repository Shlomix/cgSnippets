# ===== research/qwen2_5/infer/transformer.py — minimal patch =====
from mindspore import ops, Tensor, dtype as mstype

# --- in __init__ (add a dedicated FA for prefill; decode path untouched) ---
if getattr(self, "use_flash_attention", False):
    # Prefill on Ascend should use BNSD layout; your existing TH FA (if any) stays for decode.
    self.fa_prefill = FlashAttention(
        head_num=self.num_heads_per_partition,
        scale_value=1.0 / self.norm_factor,
        next_tokens=0,
        input_layout="BNSD",
    )
else:
    self.fa_prefill = None

# --- inside construct(...) ---

# 0) ALWAYS write current chunk to cache first (this line already exists in your file)
key_out = self.paged_attention_mgr(
    key, value, slot_mapping, batch_valid_length,
    key_cache=key_cache, value_cache=value_cache
)
query = ops.depend(query, key_out)  # keep the ordering barrier

# 1) Use q_seq_lens to decide "chunked prefill" (do not trust query.shape[1])
is_chunked = (q_seq_lens is not None) and ops.all(ops.gt(q_seq_lens, Tensor(1, mstype.int32)))

# 2) Second+ prefill: try FA once, then otherwise fall back to the original code
if (not self.is_first_iteration) and getattr(self, "use_flash_attention", False) and (self.fa_prefill is not None) and is_chunked:
    # Gather contiguous K/V from cache (they were just written by the line above)
    kc = self.paged_attention_mgr.key_cache
    vc = self.paged_attention_mgr.value_cache
    B  = int(block_tables.shape[0])
    M  = int(block_tables.shape[1])
    bs = int(kc.shape[1])                 # cache: [num_blocks, block_size, Hk]
    Hk = int(kc.shape[-1])

    flat   = ops.reshape(block_tables, (B * M,))
    k_full = ops.reshape(ops.gather(kc, flat, 0), (B, M * bs, Hk))
    v_full = ops.reshape(ops.gather(vc, flat, 0), (B, M * bs, Hk))

    # Minimal validity check so our BNSD reshape is legal (no other guards)
    nh = int(self.num_heads_per_partition)
    Hq = int(getattr(query, "shape", (0, 0, 0))[-1] or 0)
    if (Hq == Hk) and (nh > 0) and (Hq % nh == 0):
        # Convert to BNSD and run FA. Let FA derive staircase from (q_seq_lens, batch_valid_length).
        # S_cur is only used for reshape; if dynamic in your build, it’s fixed by config.
        S_cur = int(getattr(query, "shape", (0, 1, 0))[1] or 1)
        D     = Hq // nh
        KV    = M * bs

        q_bnsd = ops.transpose(ops.reshape(query,  (B, S_cur, nh, D)), (0, 2, 1, 3))
        k_bnsd = ops.transpose(ops.reshape(k_full, (B, KV,   nh, D)), (0, 2, 1, 3))
        v_bnsd = ops.transpose(ops.reshape(v_full, (B, KV,   nh, D)), (0, 2, 1, 3))

        context_layer = self.fa_prefill(
            q_bnsd, k_bnsd, v_bnsd,
            attn_mask, alibi_mask, None, None,
            q_seq_lens, batch_valid_length
        )
        return context_layer

# 3) If the one check above fails, we simply fall through to your original code paths:
#    - first-iteration branch (unchanged)
#    - original not-first-iteration path (paged/core), unchanged
# ===============================================================

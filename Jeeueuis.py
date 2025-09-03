# ===== transformer.py: minimal, static patch (no helpers returning None) =====
from mindspore import ops, Tensor, dtype as mstype
# ... keep your other imports ...

# --- in __init__ (replace single FlashAttention init with two) ---
if self.use_flash_attention:
    # prefill (S>1): BNSD kernel
    self.fa_prefill = FlashAttention(
        head_num=self.num_heads_per_partition,
        scale_value=1.0 / self.norm_factor,
        next_tokens=0,
        input_layout="BNSD",
    )
    # decode (S==1): TH kernel (leave your decode path as-is if it uses it)
    self.fa_decode = FlashAttention(
        head_num=self.num_heads_per_partition,
        scale_value=1.0 / self.norm_factor,
        next_tokens=0,
        input_layout="TH",
    )
else:
    self.fa_prefill = None
    self.fa_decode  = None

# --- small utility (pure static ints) ---
def _static_dim(self, x, i, default=1):
    shp = getattr(x, "shape", ())
    if i < len(shp) and isinstance(shp[i], int) and shp[i] is not None and shp[i] > 0:
        return int(shp[i])
    return int(default)

# --- inside construct(...), keep everything up to the manager write EXACTLY as-is ---
# key_out = self.paged_attention_mgr(key, value, slot_mapping, batch_valid_length, key_cache=..., value_cache=...)
# query   = ops.depend(query, key_out)  # ensures write happens before any compute

# ---- NEW: later-chunk prefill via FA (static, no None paths) ----
S_cur = self._static_dim(query, 1, default=1)
is_chunked = (S_cur > 1) and (q_seq_lens is not None)

if (not self.is_first_iteration) and self.use_flash_attention and is_chunked:
    kc = self.paged_attention_mgr.key_cache
    vc = self.paged_attention_mgr.value_cache
    if (kc is None) or (vc is None):
        # cache not yet ready on this call -> fall back to original paged path
        context_layer = self.paged_attention_mgr.paged_attn(
            query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=q_seq_lens
        )
        return context_layer

    # Gather contiguous KV from cache (static shapes only)
    B  = int(block_tables.shape[0])
    M  = int(block_tables.shape[1])        # compile-time
    bs = int(kc.shape[1])                  # cache: [num_blocks, block_size, Hk]
    Hk = int(kc.shape[-1])
    flat   = ops.reshape(block_tables, (B * M,))                       # (B*M,)
    k_full = ops.reshape(ops.gather(kc, flat, 0), (B, M * bs, Hk))     # (B, KV_CAP, Hk)
    v_full = ops.reshape(ops.gather(vc, flat, 0), (B, M * bs, Hk))
    KV_CAP = M * bs

    # Head compatibility check (no dynamic/types)
    nh = int(self.num_heads_per_partition)
    Hq = self._static_dim(query, 2, default=Hk)   # assume match if unknown
    # We require Hk == Hq and divisible by nh to form BNSD cleanly.
    can_fa = (Hk == Hq) and (nh > 0) and (Hq % nh == 0)

    # Optional length guard for very long KV on Ascend FA tiler
    MAX_FA_KV = 12288  # conservative; raise if your box tolerates more
    can_fa = can_fa and (KV_CAP <= MAX_FA_KV)

    if not can_fa:
        # fallback: original paged attention compute
        context_layer = self.paged_attention_mgr.paged_attn(
            query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=q_seq_lens
        )
        return context_layer

    # Convert BSH -> BNSD (pure static ints)
    Bq = self._static_dim(query, 0, default=B)
    Sq = self._static_dim(query, 1, default=S_cur)
    D  = Hq // nh
    q_bnsd = ops.transpose(ops.reshape(query,  (Bq, Sq, nh, D)), (0, 2, 1, 3))
    k_bnsd = ops.transpose(ops.reshape(k_full, (B,  KV_CAP, nh, D)), (0, 2, 1, 3))
    v_bnsd = ops.transpose(ops.reshape(v_full, (B,  KV_CAP, nh, D)), (0, 2, 1, 3))

    # Call FA for prefill; staircase is derived internally from (q_seq_lens, batch_valid_length)
    context_layer = self.fa_prefill(
        q_bnsd, k_bnsd, v_bnsd, attn_mask, alibi_mask, None, None, q_seq_lens, batch_valid_length
    )
    return context_layer

# ---- everything below stays exactly as in your file (first-iter/decoded paths) ----

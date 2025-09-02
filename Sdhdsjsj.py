# =====================================================================
# === FILE 1: mindformers/.../parallel_paged_attention_mgr.py =========
# === Add this method INSIDE class ParallelPagedAttentionMgr ==========
# =====================================================================

def materialize_prefix(self, batch_valid_length, block_tables):
    """
    Build a contiguous KV prefix from this manager's paged cache using
    a static block count M = block_tables.shape[1].

    Inputs
      batch_valid_length: (B,) int32
      block_tables:       (B, M) int32      # M = max blocks per request

    Returns
      k_pref, v_pref: (B, M*block_size, hidden_flat)
      T_max:          int32 scalar tensor = M*block_size
    """
    from mindspore import ops as P, Tensor, dtype as mstype

    kc, vc = self.key_cache, self.value_cache
    if kc is None or vc is None:
        raise RuntimeError("KV cache not initialized in ParallelPagedAttentionMgr.")

    # static sizes from shapes
    B = int(block_tables.shape[0])
    M = int(block_tables.shape[1])              # fixed, known at compile time
    block_size  = int(kc.shape[1])              # cache: [blocks, block_size, hidden_flat]
    hidden_flat = int(kc.shape[-1])

    # gather all M blocks listed for each row (static)
    flat_blk = P.Reshape()(block_tables, (B * M,))   # (B*M,)
    gk = P.Gather()(kc, flat_blk, 0)                 # (B*M, block_size, hidden_flat)
    gv = P.Gather()(vc, flat_blk, 0)

    # fold back and merge time axis (static shapes only)
    k_blocks = P.Reshape()(gk, (B, M, block_size, hidden_flat))
    v_blocks = P.Reshape()(gv, (B, M, block_size, hidden_flat))
    k_pref = P.Reshape()(k_blocks, (B, M * block_size, hidden_flat))   # (B, T_max, H)
    v_pref = P.Reshape()(v_blocks, (B, M * block_size, hidden_flat))

    T_max = Tensor(M * block_size, mstype.int32)  # kv length for mask math

    # optional numeric marker
    print(2201, M, block_size)    # 2201 = manager prefix path marker

    return k_pref, v_pref, T_max


# =====================================================================
# === FILE 2: research/qwen2_5/.../transformer.py  (attention class) ===
# === Add the imports, FA init, mask helper, probe, and FA-for-all ====
# =====================================================================

# --- [TOP OF FILE] add imports (with your others) ---
from mindformers.modules.flash_attention import FlashAttention
from mindspore import ops as P, Tensor, dtype as mstype

# --- [INSIDE __init__ of the attention class] create FlashAttention once ---
self.flash_attn = FlashAttention(
    head_num=getattr(self, "num_heads", getattr(self, "n_heads", None)),
    scale_value=getattr(self, "scale", getattr(self, "scale_value", 1.0)),
    input_layout="BSH",
    sparse_mode=0   # attn_mask: 0=keep, 1=mask
)

# --- [INSIDE THE SAME CLASS] step-wise mask helper (0=keep,1=mask) ---
def _stepwise_mask(self, L_prev_vec, S_cur, kv_len):
    """
    Step-wise causal mask for chunked prefill.
      0 = keep, 1 = mask.  Shape: (B, 1, S_cur, kv_len)
    """
    B = L_prev_vec.shape[0]
    rng_k = P.Range()(0, kv_len, 1)        # (kv_len,)
    rng_r = P.Range()(0, S_cur, 1)         # (S_cur,)

    k_idx = P.BroadcastTo((B,1,S_cur,kv_len))(P.ExpandDims()(P.ExpandDims()(rng_k, 0), 0))
    row   = P.BroadcastTo((B,1,S_cur,kv_len))(P.ExpandDims()(P.ExpandDims()(P.ExpandDims()(rng_r, -1), 0), 0))
    Lp    = P.BroadcastTo((B,1,S_cur,kv_len))(P.ExpandDims()(P.ExpandDims()(P.ExpandDims()(L_prev_vec, -1), -1), -1))

    allow = P.LessEqual()(k_idx, Lp + row)
    return P.Cast()(P.LogicalNot()(allow), mstype.uint8)

# --- [AT THE VERY TOP OF construct(...)] numbers-only probe & flags ---
_qsh   = getattr(query, "shape", None)  # query ~ [B, S_cur, H]
S_cur  = int(_qsh[1]) if (_qsh and len(_qsh) > 1 and _qsh[1] is not None) else 1

chunked_prefill = 1 if (S_cur > 1 and q_seq_lens is not None) else 0  # first + later chunks
bvl_present     = 1 if (batch_valid_length is not None) else 0

print(2102, S_cur, chunked_prefill, bvl_present)  # marker, S_cur, chunked?, bvl present?

# --- [FA FOR ALL PREFILL CHUNKS] replace only the prefill paged call ---
mgr = self.paged_attention_mgr
cache_ready = 1 if (getattr(mgr, "key_cache", None) is not None and getattr(mgr, "value_cache", None) is not None) else 0
print(2103, cache_ready, chunked_prefill)   # cache_ready vs chunked flag

if chunked_prefill == 1:
    if cache_ready == 1:
        # LATER CHUNKS: prefix + chunk
        k_pref, v_pref, T_max = mgr.materialize_prefix(batch_valid_length, block_tables)  # (B, T_max, H)
        k_full = P.Concat(axis=1)((k_pref, key))   # (B, T_max + S_cur, H)
        v_full = P.Concat(axis=1)((v_pref, value))
        kv_len = T_max + Tensor(S_cur, mstype.int32)
        step_mask = self._stepwise_mask(batch_valid_length, S_cur, kv_len=kv_len)
        context = self.flash_attn(query, k_full, v_full, attn_mask=step_mask)
        print(3001, 1)  # FA (later-chunk)
    else:
        # FIRST CHUNK: no prefix; triangular within chunk
        kv_len = Tensor(S_cur, mstype.int32)
        step_mask = self._stepwise_mask(batch_valid_length, S_cur, kv_len=kv_len)
        context = self.flash_attn(query, key, value, attn_mask=step_mask)
        print(3002, 1)  # FA (first-chunk)

    # IMPORTANT: keep cache updated so decode works
    # Requires slot_mapping (same one used by paged_attn); adjust the name if different.
    if cache_ready == 1:
        mgr.reshape_and_cache(key, value, mgr.key_cache, mgr.value_cache, slot_mapping)
    else:
        # if caches were not allocated yet, skip safely; they will exist by next chunk
        if getattr(mgr, "key_cache", None) is not None:
            mgr.reshape_and_cache(key, value, mgr.key_cache, mgr.value_cache, slot_mapping)
else:
    # decode or non-chunked path -> original paged attention
    context = mgr.paged_attn(
        query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=q_seq_lens
    )

# ... then continue exactly as before using/returning `context`.

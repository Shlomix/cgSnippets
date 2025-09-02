# =====================================================================
# === FILE 1: mindformers/.../parallel_paged_attention_mgr.py =========
# === Add this method INSIDE class ParallelPagedAttentionMgr ==========
# === (static-shape, no layer_idx; avoids engine init failures) =======
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

    # --- static sizes from shapes (compile-time ints) ---
    B = int(block_tables.shape[0])
    M = int(block_tables.shape[1])              # fixed, known at compile time
    block_size = int(kc.shape[1])               # cache layout: [blocks, block_size, hidden_flat]
    hidden_flat = int(kc.shape[-1])

    # --- gather all M blocks listed for each row (static) ---
    # block_tables: (B, M) -> flatten to (B*M,)
    flat_blk = P.Reshape()(block_tables, (B * M,))
    # Gather along blocks axis (axis=0)
    gk = P.Gather()(kc, flat_blk, 0)            # (B*M, block_size, hidden_flat)
    gv = P.Gather()(vc, flat_blk, 0)

    # --- fold back and merge time axis (static shapes only) ---
    # reshape to (B, M, block_size, hidden_flat) then merge M*block_size
    k_blocks = P.Reshape()(gk, (B, M, block_size, hidden_flat))
    v_blocks = P.Reshape()(gv, (B, M, block_size, hidden_flat))

    k_pref = P.Reshape()(k_blocks, (B, M * block_size, hidden_flat))   # (B, T_max, H)
    v_pref = P.Reshape()(v_blocks, (B, M * block_size, hidden_flat))

    # kv length as Tensor for downstream ops/masks
    T_max = Tensor(M * block_size, mstype.int32)

    # optional tiny numeric marker (keep while testing)
    print(2201, M, block_size)    # 2201 = manager prefix path marker

    return k_pref, v_pref, T_max


# =====================================================================
# === FILE 2: research/qwen2_5/.../transformer.py  (attention class) ===
# === Add the imports, FA init, mask helper, probe, and FA branch  ====
# =====================================================================

# --- [TOP OF FILE] add imports (with your other imports) ---
from mindformers.modules.flash_attention import FlashAttention
from mindspore import ops as P, Tensor, dtype as mstype

# --- [INSIDE __init__ of the attention class] create FlashAttention once ---
self.flash_attn = FlashAttention(
    head_num=getattr(self, "num_heads", getattr(self, "n_heads", None)),
    scale_value=getattr(self, "scale", getattr(self, "scale_value", 1.0)),
    input_layout="BSH",
    sparse_mode=0   # attn_mask uses 0=keep, 1=mask
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
    return P.Cast()(P.LogicalNot()(allow), mstype.uint8)  # 0 keep, 1 mask

# --- [AT THE VERY TOP OF construct(...)] numeric-only probe (no strings) ---
# choose a tensor here that is [B, S_cur, H]; usually this is `query`
_qsh   = getattr(query, "shape", None)  # query expected: [B, S_cur, H]
S_cur  = int(_qsh[1]) if (_qsh and len(_qsh) > 1 and _qsh[1] is not None) else 1
prefill = 1 if (S_cur > 1) else 0
qsl_set = 1 if (q_seq_lens is not None) else 0
bvl_set = 1 if (batch_valid_length is not None) else 0
later_prefill = 1 if (prefill and qsl_set and bvl_set) else 0
print(2101, S_cur, prefill, qsl_set, bvl_set, later_prefill)  # one numeric line

# --- [REPLACE ONLY THE PREFILL PAGED CALL with this guarded FA branch] ---
if later_prefill == 1:
    # 1) materialize a contiguous prefix from ALL blocks in block_tables (static M)
    k_pref, v_pref, T_max = self.paged_attention_mgr.materialize_prefix(
        batch_valid_length, block_tables
    )  # (B, T_max, H_flat), T_max = M*block_size

    # 2) concat prefix with current chunk's K/V along time axis -> [prefix + chunk]
    # If your key/value are [B,S,heads,head_dim], reshape k_pref/v_pref accordingly before Concat.
    k_full = P.Concat(axis=1)((k_pref, key))     # (B, T_max + S_cur, H_flat)
    v_full = P.Concat(axis=1)((v_pref, value))

    # 3) step-wise mask for kv_len = T_max + S_cur (0=keep, 1=mask)
    kv_len    = T_max + Tensor(S_cur, mstype.int32)
    step_mask = self._stepwise_mask(batch_valid_length, S_cur, kv_len=kv_len)

    # 4) FlashAttention on [prefix + chunk]
    context = self.flash_attn(query, k_full, v_full, attn_mask=step_mask)

    # tiny numeric confirmation
    print(3001, 1)
else:
    # first prefill (or non-chunked) -> keep existing paged attention
    context = self.paged_attention_mgr.paged_attn(
        query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=q_seq_lens
    )

# ...then continue exactly as before using/returning `context`.

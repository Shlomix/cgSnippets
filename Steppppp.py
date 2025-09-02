# =====================================================================
# === FILE 1: mindformers/.../parallel_paged_attention_mgr.py =========
# === Add this method INSIDE class ParallelPagedAttentionMgr ==========
# === (no layer_idx; caches are per-layer in this manager) ============
# =====================================================================

def materialize_prefix(self, batch_valid_length, block_tables):
    """
    Build a contiguous KV prefix from this manager's paged cache.

    Inputs
      batch_valid_length: (B,) int32          # cached prefix tokens per request
      block_tables:       (B, max_blocks) int32   # block IDs per request

    Returns
      k_pref, v_pref: (B, T_max, hidden_flat)     # contiguous prefix (padded to same T_max across batch)
      T_max:          int32 scalar tensor         # T_max = (#blocks taken) * block_size
    """
    from mindspore import ops as P, Tensor, dtype as mstype

    kc, vc = self.key_cache, self.value_cache
    if kc is None or vc is None:
        raise RuntimeError("KV cache not initialized in ParallelPagedAttentionMgr.")

    B = batch_valid_length.shape[0]

    # Cache layout in this manager: [blocks, block_size, ...]
    blocks_axis, bs_axis = 0, 1
    block_size = int(kc.shape[bs_axis])
    bs_i32 = Tensor(block_size, mstype.int32)

    # Take the same number of blocks for all rows: max(full_blocks) + 1 (to cover a partial)
    n_full   = P.FloorDiv()(batch_valid_length, bs_i32)     # (B,)
    max_full = P.ReduceMax(False)(n_full)                   # scalar
    take     = max_full + Tensor(1, mstype.int32)           # include partial

    # block_tables[:, :take]
    rng   = P.Range()(0, take, 1)                           # (take,)
    rng   = P.ExpandDims()(rng, 0)                          # (1, take)
    idx2  = P.BroadcastTo((B, take))(rng)                   # (B, take)
    blk_ids = P.GatherD()(block_tables, 1, idx2)            # (B, take)

    # Gather blocks from cache along blocks axis.
    flat_blk = P.Reshape()(blk_ids, (-1,))                  # (B*take,)
    gk = P.Gather()(kc, flat_blk, blocks_axis)              # (B*take, block_size, hidden_flat...)
    gv = P.Gather()(vc, flat_blk, blocks_axis)

    # Fold back and merge time axis -> contiguous prefix
    take_s   = P.Reshape()(take, ())                        # scalar
    k_blocks = P.Reshape()(gk, (B, take_s, block_size, -1))
    v_blocks = P.Reshape()(gv, (B, take_s, block_size, -1))

    T_max = take_s * bs_i32                                 # scalar int32
    k_pref = P.Reshape()(k_blocks, (B, T_max, -1))          # (B, T_max, hidden_flat)
    v_pref = P.Reshape()(v_blocks, (B, T_max, -1))
    return k_pref, v_pref, T_max


# =====================================================================
# === FILE 2: research/qwen2_5/.../transformer.py  (your attention) ===
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
    sparse_mode=0   # we pass a dense mask: 0=keep, 1=mask
)

# --- [INSIDE THE SAME CLASS] step-wise mask helper (0=keep,1=mask) ---
def _stepwise_mask(self, L_prev_vec, S_cur, kv_len=None):
    """
    Step-wise causal mask for chunked prefill.  Shape: (B, 1, S_cur, kv_len)
    0 = keep, 1 = mask.
    """
    B = L_prev_vec.shape[0]
    if kv_len is None:
        kv_len = P.ReduceMax(False)(L_prev_vec) + Tensor(S_cur, mstype.int32)

    rng_k = P.Range()(0, kv_len, 1)        # (kv_len,)
    rng_r = P.Range()(0, S_cur, 1)         # (S_cur,)

    k_idx = P.BroadcastTo((B,1,S_cur,kv_len))(P.ExpandDims()(P.ExpandDims()(rng_k, 0), 0))
    row   = P.BroadcastTo((B,1,S_cur,kv_len))(P.ExpandDims()(P.ExpandDims()(P.ExpandDims()(rng_r, -1), 0), 0))
    Lp    = P.BroadcastTo((B,1,S_cur,kv_len))(P.ExpandDims()(P.ExpandDims()(P.ExpandDims()(L_prev_vec, -1), -1), -1))

    allow = P.LessEqual()(k_idx, Lp + row)
    return P.Cast()(P.LogicalNot()(allow), mstype.uint8)

# --- [AT THE VERY TOP OF construct(...)] keep a tiny numeric probe (Step-2) ---
_qsh   = getattr(query, "shape", None)          # query expected: [B, S_cur, H]
S_cur  = int(_qsh[1]) if (_qsh and len(_qsh) > 1 and _qsh[1] is not None) else 1

prefill = 1 if (S_cur > 1) else 0
qsl_set = 1 if (q_seq_lens is not None) else 0
bvl_set = 1 if (batch_valid_length is not None) else 0
later_prefill = 1 if (prefill and qsl_set and bvl_set) else 0

# one numeric line: marker, S_cur, prefill?, q_seq_lens?, bvl?, later_prefill?
print(2101, S_cur, prefill, qsl_set, bvl_set, later_prefill)

# --- [REPLACE ONLY THE PREFILL PAGED CALL with this guarded FA branch] ---
if later_prefill == 1:
    # 1) contiguous prefix from the manager (no layer_idx needed in this repo)
    k_pref, v_pref, T_max = self.paged_attention_mgr.materialize_prefix(
        batch_valid_length, block_tables
    )  # (B, T_max, hidden_flat)

    # 2) concat prefix with this chunk's K/V along time axis -> [prefix + chunk]
    k_full = P.Concat(axis=1)((k_pref, key))     # (B, T_max + S_cur, hidden_flat)
    v_full = P.Concat(axis=1)((v_pref, value))

    # 3) step-wise mask for kv_len = T_max + S_cur (0=keep, 1=mask)
    kv_len    = T_max + Tensor(S_cur, mstype.int32)
    step_mask = self._stepwise_mask(batch_valid_length, S_cur, kv_len=kv_len)

    # 4) FlashAttention over [prefix + chunk]
    context = self.flash_attn(query, k_full, v_full, attn_mask=step_mask)

    # tiny numeric confirmation
    print(3001, 1)
else:
    # original paged prefill / decode path (unchanged)
    context = self.paged_attention_mgr.paged_attn(
        query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=q_seq_lens
    )

# ...continue exactly as before using/returning `context`.

# =====================================================================
# === FILE: mindformers/.../parallel_paged_attention_mgr.py ============
# === Add this method INSIDE class ParallelPagedAttentionMgr ===========
# =====================================================================

def materialize_prefix(self, layer_idx, batch_valid_length, block_tables):
    """
    Build a contiguous KV prefix from the paged cache.

    Inputs
      layer_idx: int          # which layer cache to read (ignored if cache is per-layer instance)
      batch_valid_length: (B,) int32
      block_tables:       (B, max_blocks) int32

    Returns
      k_pref, v_pref: (B, T_max, hidden_flat)    # contiguous prefix per batch (padded to same T_max)
      T_max:          int32 scalar tensor        # T_max = (#blocks taken) * block_size
    """
    from mindspore import ops as P, Tensor, dtype as mstype

    kc, vc = self.key_cache, self.value_cache
    if kc is None or vc is None:
        raise RuntimeError("KV cache not initialized in ParallelPagedAttentionMgr.")

    B = batch_valid_length.shape[0]
    shp = kc.shape

    # ---- detect layout & axes (two common layouts supported) ----
    #   [layers, blocks, block_size, ...]  -> slice layer, then blocks_axis=0, bs_axis=1
    #   [blocks, block_size, ...]          -> blocks_axis=0, bs_axis=1
    if len(shp) >= 3 and shp[0] > 1 and shp[1] >= 1:
        kc = kc[layer_idx]
        vc = vc[layer_idx]
        blocks_axis, bs_axis = 0, 1
    else:
        blocks_axis, bs_axis = 0, 1

    block_size = int(kc.shape[bs_axis])
    bs_i32 = Tensor(block_size, mstype.int32)

    # ---- choose how many blocks to fetch for all rows (max + 1 partial) ----
    n_full   = P.FloorDiv()(batch_valid_length, bs_i32)        # (B,)
    max_full = P.ReduceMax(False)(n_full)                      # scalar
    take     = max_full + Tensor(1, mstype.int32)              # include partial block

    # ---- gather block IDs per row: block_tables[:, :take] ----
    rng   = P.Range()(0, take, 1)                              # (take,)
    rng   = P.ExpandDims()(rng, 0)                             # (1, take)
    idx2  = P.BroadcastTo((B, take))(rng)                      # (B, take)
    blk_ids = P.GatherD()(block_tables, 1, idx2)               # (B, take)

    # ---- fetch blocks along the blocks axis ----
    flat_blk = P.Reshape()(blk_ids, (-1,))                     # (B*take,)
    gk = P.Gather()(kc, flat_blk, blocks_axis)                 # (B*take, block_size, hidden_flat...)
    gv = P.Gather()(vc, flat_blk, blocks_axis)

    # ---- fold back and merge time axis -> contiguous prefix ----
    take_s   = P.Reshape()(take, ())                           # scalar
    k_blocks = P.Reshape()(gk, (B, take_s, block_size, -1))
    v_blocks = P.Reshape()(gv, (B, take_s, block_size, -1))

    T_max = take_s * bs_i32                                    # scalar int32
    k_pref = P.Reshape()(k_blocks, (B, T_max, -1))             # (B, T_max, hidden_flat)
    v_pref = P.Reshape()(v_blocks, (B, T_max, -1))
    return k_pref, v_pref, T_max


# =====================================================================
# === FILE: research/qwen2_5/.../transformer.py (your attention class)
# === Add/modify the following in that file/class =====================
# =====================================================================

# --- [TOP OF FILE] add imports (with the others) ---
from mindformers.modules.flash_attention import FlashAttention
from mindspore import ops as P, Tensor, dtype as mstype

# --- [INSIDE __init__ of the attention class] create FA once ---
self.flash_attn = FlashAttention(
    head_num=getattr(self, "num_heads", getattr(self, "n_heads", None)),
    scale_value=getattr(self, "scale", getattr(self, "scale_value", 1.0)),
    input_layout="BSH",
    sparse_mode=0   # we pass a dense mask (0=keep, 1=mask)
)

# --- [INSIDE THE SAME CLASS] add this small helper once ---
def _stepwise_mask(self, L_prev_vec, S_cur, kv_len=None):
    """
    Step-wise causal mask for chunked prefill.
      0 = keep, 1 = mask.  Shape: (B, 1, S_cur, kv_len)
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
    return P.Cast()(P.LogicalNot()(allow), mstype.uint8)  # 0 keep, 1 mask

# --- [AT THE VERY TOP OF construct(...)] keep a numeric-only Step-2 probe ---
# choose a tensor here that is [B, S_cur, H]; usually this is `query`
_qsh   = getattr(query, "shape", None)
S_cur  = int(_qsh[1]) if (_qsh and len(_qsh) > 1 and _qsh[1] is not None) else 1
prefill = 1 if (S_cur > 1) else 0
qsl_set = 1 if (q_seq_lens is not None) else 0
bvl_set = 1 if (batch_valid_length is not None) else 0
later_prefill = 1 if (prefill and qsl_set and bvl_set) else 0
print(2101, S_cur, prefill, qsl_set, bvl_set, later_prefill)  # one numeric line

# --- [REPLACE ONLY THE PREFILL PAGED CALL inside construct(...)] ---
# OLD (example):
# context = self.paged_attention_mgr.paged_attn(
#     query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=q_seq_lens
# )

# NEW:
if later_prefill == 1:
    # 1) contiguous prefix from the cache manager
    layer_idx = int(getattr(self, "layer_idx", getattr(self, "layer_id", 0)))
    k_pref, v_pref, T_max = self.paged_attention_mgr.materialize_prefix(
        layer_idx, batch_valid_length, block_tables
    )  # (B, T_max, hidden_flat)

    # 2) concat prefix with current chunk's K/V along time axis -> [prefix + chunk]
    k_full = P.Concat(axis=1)((k_pref, key))     # (B, T_max + S_cur, hidden_flat)
    v_full = P.Concat(axis=1)((v_pref, value))

    # 3) step-wise mask for kv_len = T_max + S_cur (0=keep,1=mask)
    kv_len    = T_max + Tensor(S_cur, mstype.int32)
    step_mask = self._stepwise_mask(batch_valid_length, S_cur, kv_len=kv_len)

    # 4) FlashAttention on [prefix + chunk]
    context = self.flash_attn(query, k_full, v_full, attn_mask=step_mask)

    # numeric confirmation for FA branch
    print(3001, 1)
else:
    # first prefill (or non-chunked) -> keep existing paged attention
    context = self.paged_attention_mgr.paged_attn(
        query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=q_seq_lens
    )

# (then continue exactly as your original construct did, returning `context`)
# =====================================================================

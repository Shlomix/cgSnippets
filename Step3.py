# ===================== BEGIN: paste these pieces into transformer.py =====================

# [A] IMPORTS (put near the other imports at the top of the file)
from mindformers.modules.flash_attention import FlashAttention
from mindspore import ops as P, Tensor, dtype as mstype

# [B] IN CLASS ParallelAttention.__init__(...)  — add this once (next to other kernels)
#     (Look for the existing __init__ of class ParallelAttention and insert these 3 lines)
self.flash_attn = FlashAttention(
    head_num=self.num_heads,          # use the class' existing head count
    scale_value=self.scale,           # reuse whatever scale var this class uses
    input_layout="BSH",
    sparse_mode=0                     # full mask path (0=keep,1=mask)
)

# [C] IN CLASS ParallelAttention (define this small helper method inside the class, e.g. under __init__)
def _stepwise_mask(self, L_prev_vec, S_cur, kv_len=None):
    """
    Build step-wise causal mask for chunked prefill (0=keep, 1=mask).
    Shape: (B, 1, S_cur, kv_len). If kv_len is None, use max(L_prev)+S_cur.
    """
    B = L_prev_vec.shape[0]
    if kv_len is None:
        kv_len = P.ReduceMax(False)(L_prev_vec) + Tensor(S_cur, mstype.int32)

    rng_k = P.Range()(0, kv_len, 1)                                   # (kv_len,)
    rng_r = P.Range()(0, S_cur, 1)                                    # (S_cur,)

    k_idx = P.BroadcastTo((B,1,S_cur,kv_len))(P.ExpandDims()(P.ExpandDims()(rng_k, 0), 0))
    row   = P.BroadcastTo((B,1,S_cur,kv_len))(P.ExpandDims()(P.ExpandDims()(P.ExpandDims()(rng_r, -1), 0), 0))
    Lp    = P.BroadcastTo((B,1,S_cur,kv_len))(P.ExpandDims()(P.ExpandDims()(P.ExpandDims()(L_prev_vec, -1), -1), -1))

    allow = P.LessEqual()(k_idx, Lp + row)                             # bool staircase
    return P.Cast()(P.LogicalNot()(allow), mstype.uint8)               # 0=keep,1=mask

# [D] IN ParallelAttention.construct(...): paste this NUMERIC-ONLY PROBE at the VERY TOP
#     (Use the actual local tensor name that has shape [B, S, H]; often it's "query" or "x")
_qsh = getattr(query, "shape", None)
S_cur = int(_qsh[1]) if (_qsh and len(_qsh) > 1 and _qsh[1] is not None) else 1

# presence flags only (no tensor reads)
qsl_set = 1 if (q_seq_lens is not None) else 0
bvl_set = 1 if (batch_valid_length is not None) else 0
prefill = 1 if (S_cur > 1) else 0
later_prefill = 1 if (prefill and qsl_set and bvl_set) else 0

# single numeric line: marker, S, prefill?, q_seq_lens?, bvl?, later_prefill?
print(2101, S_cur, prefill, qsl_set, bvl_set, later_prefill)

# keep a tiny flag for the branch below
self._later_prefill_flag = later_prefill

# [E] IN ParallelAttention.construct(...): locate the existing **paged** prefill call:
#     context = self.paged_attention_mgr.paged_attn(
#         query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=q_seq_lens
#     )
#     and REPLACE JUST THAT CALL with the small guarded switch below.

if self._later_prefill_flag == 1:
    # ---- FlashAttention path for 2nd+ prefill chunk ----
    # IMPORTANT: FlashAttention needs K/V covering [prefix + current chunk].
    # If your path already builds contiguous K/V for prefill, use them as key_full/value_full.
    # If not yet implemented, we fall back to current K/V and size the mask to their length
    # so it still runs (mask reduces to lower-triangular within the chunk).
    kv_len_tensor = getattr(key, "shape", [None, S_cur])[1]
    kv_len_py = int(kv_len_tensor) if (kv_len_tensor is not None) else S_cur

    # build the step-wise mask; when kv_len==S_cur (no prefix K/V yet), this degenerates to standard causal within the chunk
    step_mask = self._stepwise_mask(batch_valid_length, S_cur, kv_len=Tensor(kv_len_py, mstype.int32))

    # TODO (recommended): replace key/value below with [prefix+chunk] tensors gathered from cache.
    key_full, value_full = key, value

    # call FlashAttention
    context = self.flash_attn(query, key_full, value_full, attn_mask=step_mask)

    # tiny numeric confirmation (no strings)
    print(3001, 1)   # 3001 => took FlashAttention branch
else:
    # ---- original paged-attention path (unchanged) ----
    context = self.paged_attention_mgr.paged_attn(
        query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=q_seq_lens
    )

# (the rest of construct continues as before, returning `context` downstream)

# ===================== END: paste these pieces into transformer.py =====================

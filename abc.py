# ========================= transformer.py PATCH =========================
# 1) Add (or ensure) these imports near the top of the file
from mindspore import ops, Tensor, dtype as mstype

# 2) INSIDE YOUR ATTENTION CLASS, add two tiny helpers (anywhere after __init__)

def _kv_from_cache(self, block_tables, fallback_key=None, fallback_value=None):
    """
    Read a contiguous K/V view from the paged cache.
    If the cache isn't ready yet, fall back to the current chunk tensors.
    Returns:
      k_full, v_full: (B, T_cap, H)
      kv_len_cap    : Tensor[int32] = T_cap
    """
    kc = getattr(self.paged_attention_mgr, "key_cache", None)
    vc = getattr(self.paged_attention_mgr, "value_cache", None)

    # Fallback: cache not ready -> use current chunk only (triangular FA)
    if kc is None or vc is None:
        if fallback_key is None or fallback_value is None:
            raise RuntimeError("KV cache not initialized and no fallback K/V provided.")
        S = int(getattr(fallback_key, "shape", (0, 1, 0))[1] or 1)
        return fallback_key, fallback_value, Tensor(S, mstype.int32)

    # Normal path: gather contiguous K/V from cache (static shapes only)
    B  = int(block_tables.shape[0])
    M  = int(block_tables.shape[1])          # compile-time constant
    bs = int(kc.shape[1])                    # cache layout: [num_blocks, block_size, H]
    H  = int(kc.shape[-1])

    flat = ops.reshape(block_tables, (B * M,))   # (B*M,)
    gk   = ops.gather(kc, flat, 0)               # (B*M, bs, H)
    gv   = ops.gather(vc, flat, 0)

    k_blocks = ops.reshape(gk, (B, M, bs, H))
    v_blocks = ops.reshape(gv, (B, M, bs, H))
    k_full   = ops.reshape(k_blocks, (B, M * bs, H))  # (B, T_cap, H)
    v_full   = ops.reshape(v_blocks, (B, M * bs, H))
    return k_full, v_full, Tensor(M * bs, mstype.int32)


def _stepwise_mask(self, L_prev_vec, S_cur, kv_len):
    """
    Staircase causal mask for chunked prefill.
    Output: uint8 mask (B, 1, S_cur, kv_len).  0 = keep, 1 = mask.
    (Uses implicit broadcasting; no BroadcastTo with Tensor in shape.)
    """
    if L_prev_vec.dtype != mstype.int32:
        L_prev_vec = ops.cast(L_prev_vec, mstype.int32)

    rng_k = ops.Range()(Tensor(0, mstype.int32), kv_len, Tensor(1, mstype.int32))                       # (kv_len,)
    rng_r = ops.Range()(Tensor(0, mstype.int32), Tensor(S_cur, mstype.int32), Tensor(1, mstype.int32))  # (S_cur,)

    k_idx = ops.expand_dims(ops.expand_dims(ops.expand_dims(rng_k, 0), 0), 0)   # (1,1,1,kv_len)
    row   = ops.expand_dims(ops.expand_dims(ops.expand_dims(rng_r, 0), 0), -1)  # (1,1,S_cur,1)
    Lp    = ops.expand_dims(ops.expand_dims(ops.expand_dims(L_prev_vec, 1), 1), 1)  # (B,1,1,1)

    allow = ops.less_equal(k_idx, Lp + row)                      # -> (B,1,S_cur,kv_len)
    return ops.cast(ops.logical_not(allow), mstype.uint8)        # 0=keep, 1=mask


# 3) INSIDE construct(...), DO NOT change code ABOVE the manager write.
# You already have something like:
#   key_out = self.paged_attention_mgr(key, value, slot_mapping, batch_valid_length, key_cache=..., value_cache=...)
#   query   = ops.depend(query, key_out)  # ensures the write happens before compute

# 4) RIGHT AFTER the ops.depend(...) line, insert this small early-exit for later prefill chunks with FA:

S_cur = int(getattr(query, "shape", (0, 1, 0))[1] or 1)
is_chunked = (S_cur > 1) and (q_seq_lens is not None)

if (not self.is_first_iteration) and getattr(self, "use_flash_attention", True) and is_chunked:
    # Read a contiguous K/V view from cache; if somehow not ready, fall back to this chunk
    k_full, v_full, kv_len_cap = self._kv_from_cache(
        block_tables, fallback_key=key, fallback_value=value
    )

    # Build staircase mask so token r can see up to L_prev + r keys
    step_mask = self._stepwise_mask(batch_valid_length, S_cur, kv_len_cap)

    # FlashAttention over cached K/V (use same wrapper/signature as in your first-iteration FA)
    context_layer = self.flash_attention(
        query, k_full, v_full, step_mask, alibi_mask,
        None, None, q_seq_lens, batch_valid_length
    )
    return context_layer

# 5) Leave the rest of your original code exactly as-is:
# if self.is_first_iteration:
#     if self.use_flash_attention:  # FA triangular (your existing call)
#         ...
#     else:                         # Simple/core attention
#         ...
# else:
#     # original not-first-iteration path (paged/simple), unchanged
#     ...
# ======================================================================

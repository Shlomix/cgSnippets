# ========================= transformer.py PATCH =========================
# --- 1) Add (or ensure) these imports near the top of the file ---
from mindspore import ops, Tensor, dtype as mstype
# (Keep any existing imports you already have.)

# --- 2) INSIDE YOUR ATTENTION CLASS, add these two helpers anywhere (e.g., after __init__) ---

def _kv_from_cache(self, block_tables):
    """
    Read a contiguous K/V view from the paged cache (no manager change needed).
    Returns:
      k_full, v_full: (B, M*block_size, H)
      kv_len_cap    : Tensor[int32] = M*block_size   (upper bound for kv length)
    """
    kc = self.paged_attention_mgr.key_cache
    vc = self.paged_attention_mgr.value_cache
    if kc is None or vc is None:
        raise RuntimeError("KV cache not initialized yet (first prefill hasn't written).")

    B = int(block_tables.shape[0])
    M = int(block_tables.shape[1])   # compile-time constant
    bs = int(kc.shape[1])            # cache layout: [num_blocks, block_size, H]
    H  = int(kc.shape[-1])

    flat = ops.reshape(block_tables, (B * M,))  # (B*M,)
    gk = ops.gather(kc, flat, 0)                # (B*M, bs, H)
    gv = ops.gather(vc, flat, 0)

    k_blocks = ops.reshape(gk, (B, M, bs, H))
    v_blocks = ops.reshape(gv, (B, M, bs, H))
    k_full   = ops.reshape(k_blocks, (B, M * bs, H))   # (B, kv_len_cap, H)
    v_full   = ops.reshape(v_blocks, (B, M * bs, H))

    kv_len_cap = Tensor(M * bs, mstype.int32)
    return k_full, v_full, kv_len_cap


def _stepwise_mask(self, L_prev_vec, S_cur, kv_len):
    """
    Staircase causal mask for chunked prefill.
    Output: uint8 mask with shape (B, 1, S_cur, kv_len).  0 = keep, 1 = mask.
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


# --- 3) INSIDE construct(...): keep everything ABOVE your screenshot exactly as-is ---
# (i.e., you already have:)
#   key_out = self.paged_attention_mgr(key, value, slot_mapping, batch_valid_length, key_cache=..., value_cache=...)
#   query   = ops.depend(query, key_out)   # ensures the cache write happens before compute
#   if self.is_first_iteration:
#       if self.use_flash_attention:
#           context_layer = self.flash_attention(query, key, value, attn_mask, alibi_mask, None, None, q_seq_lens, batch_valid_length)
#       else:
#           ... simple/core attention ...
#   else:
#       # << we insert the tiny FA path here, then fall through to your original code if it doesn't trigger >>

# --- 4) At the VERY START of the `else:` (not-first-iteration) block, insert this early-exit: ---

S_cur = int(getattr(query, "shape", (None, 1, None))[1] or 1)
is_chunked = (S_cur > 1) and (q_seq_lens is not None)
# Optional minimal probe (numbers only):
# print(99001, S_cur, int(is_chunked), int(self.use_flash_attention))

if self.use_flash_attention and is_chunked:
    # Cache already contains [prefix + current chunk] because the write above just ran.
    k_full, v_full, kv_len_cap = self._kv_from_cache(block_tables)

    # Step-wise mask so token r can see up to L_prev + r keys
    step_mask = self._stepwise_mask(batch_valid_length, S_cur, kv_len_cap)

    # FlashAttention over the cached K/V.
    # Use the same wrapper/signature you already call in first iteration.
    context_layer = self.flash_attention(
        query, k_full, v_full, step_mask, alibi_mask, None, None, q_seq_lens, batch_valid_length
    )
    return context_layer

# ... leave the rest of your original NOT-FIRST-ITERATION code exactly as-is ...
# (simple/core attention path or whatever your file does today)
# ======================================================================

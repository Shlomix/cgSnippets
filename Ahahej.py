def _stepwise_mask(self, L_prev_vec, S_cur, kv_len):
    """
    Step-wise causal mask (0=keep, 1=mask) with implicit broadcasting.
    L_prev_vec: (B,) int32    # prefix lengths already in cache
    S_cur:      python int    # current chunk length
    kv_len:     Tensor[int32] # total K/V length to compare against
    Returns: (B, 1, S_cur, kv_len) uint8
    """
    # ensure types
    from mindspore import ops, Tensor, dtype as mstype
    if L_prev_vec.dtype != mstype.int32:
        L_prev_vec = ops.cast(L_prev_vec, mstype.int32)

    # dynamic ranges (JIT-safe)
    Range = ops.Range()
    rng_k = Range(Tensor(0, mstype.int32), kv_len, Tensor(1, mstype.int32))             # (kv_len,)
    rng_r = Range(Tensor(0, mstype.int32), Tensor(S_cur, mstype.int32), Tensor(1, mstype.int32))  # (S_cur,)

    # expand without BroadcastTo; rely on implicit broadcasting in compare
    # shapes: (1,1,1,kv_len), (1,1,S_cur,1), (B,1,1,1)
    k_idx = ops.expand_dims(ops.expand_dims(ops.expand_dims(rng_k, 0), 0), 0)
    row   = ops.expand_dims(ops.expand_dims(ops.expand_dims(rng_r, 0), 0), -1)
    Lp    = ops.expand_dims(ops.expand_dims(ops.expand_dims(L_prev_vec, 1), 1), 1)

    # allow if key index <= L_prev + row
    allow = ops.less_equal(k_idx, Lp + row)                 # (B,1,S_cur,kv_len)
    return ops.cast(ops.logical_not(allow), mstype.uint8)   # 0=keep, 1=mask

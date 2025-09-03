# ---- small helpers assumed available ----
# _stepwise_mask(L_prev_vec, S_cur, kv_len)  # the implicit-broadcast version you installed
# mgr = self.paged_attention_mgr
# tensors already computed above: query, key, value, batch_valid_length, block_tables, slot_mapping, q_seq_lens

# decide flags (numbers-only print stays JIT-friendly)
S_cur = int(getattr(query, "shape", (None, 1, None))[1] or 1)
chunked_prefill = 1 if (S_cur > 1 and q_seq_lens is not None) else 0
print(2104, S_cur, chunked_prefill, int(self.is_first_iteration))

USE_FA = int(getattr(self, "use_flash_attention", 1))  # set once in __init__ if you want

if self.is_first_iteration:
    # FIRST CHUNK (prefill). If FA is enabled, use FA (triangular over the chunk).
    if USE_FA and chunked_prefill:
        kv_len = Tensor(S_cur, mstype.int32)
        step_mask = self._stepwise_mask(batch_valid_length, S_cur, kv_len)  # 0=keep,1=mask
        context = self.flash_attn(query, key, value, attn_mask=step_mask)
        print(3002, S_cur)  # FA first chunk
        # write this chunk to cache AFTER FA so the next chunk sees it as prefix
        mgr.reshape_and_cache(key, value, mgr.key_cache, mgr.value_cache, slot_mapping)
    else:
        # original paged path: stock flow writes then computes
        mgr.reshape_and_cache(key, value, mgr.key_cache, mgr.value_cache, slot_mapping)
        context = mgr.paged_attn(query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=q_seq_lens)
else:
    # NOT FIRST ITERATION
    if USE_FA and chunked_prefill:
        # later prefill chunk -> FA over [prefix + chunk]
        # contiguous prefix from cache (static-shape helper in the manager)
        k_pref, v_pref, T_max = mgr.materialize_prefix(batch_valid_length, block_tables)  # (B, T_prev, H)
        k_full = ops.concat((k_pref, key), axis=1)  # [prefix + chunk]
        v_full = ops.concat((v_pref, value), axis=1)
        kv_len = T_max + Tensor(S_cur, mstype.int32)
        step_mask = self._stepwise_mask(batch_valid_length, S_cur, kv_len)
        context = self.flash_attn(query, k_full, v_full, attn_mask=step_mask)
        print(3001, S_cur)  # FA later chunk
        # write this chunk to cache AFTER FA
        mgr.reshape_and_cache(key, value, mgr.key_cache, mgr.value_cache, slot_mapping)
    else:
        # fallback: paged prefill or decode (stock)
        # stock prefill writes before compute; decode path in your tree likely writes too—keep as is.
        mgr.reshape_and_cache(key, value, mgr.key_cache, mgr.value_cache, slot_mapping) if S_cur > 1 else None
        context = mgr.paged_attn(query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=q_seq_lens)

# in ParallelPagedAttentionMgr (unchanged from your last working version)
def materialize_prefix(self, batch_valid_length, block_tables):
    from mindspore import ops as _ops, Tensor as _T, dtype as _dt
    kc, vc = self.key_cache, self.value_cache
    B, M = int(block_tables.shape[0]), int(block_tables.shape[1])
    bs, H = int(kc.shape[1]), int(kc.shape[-1])
    flat = _ops.reshape(block_tables, (B*M,))
    gk = _ops.gather(kc, flat, 0); gv = _ops.gather(vc, flat, 0)
    k_blocks = _ops.reshape(gk, (B, M, bs, H)); v_blocks = _ops.reshape(gv, (B, M, bs, H))
    k_pref = _ops.reshape(k_blocks, (B, M*bs, H)); v_pref = _ops.reshape(v_blocks, (B, M*bs, H))
    T_max = _T(M*bs, _dt.int32)
    return k_pref, v_pref, T_max


def _stepwise_mask(self, L_prev_vec, S_cur, kv_len):
    from mindspore import ops, Tensor, dtype as mstype
    if L_prev_vec.dtype != mstype.int32:
        L_prev_vec = ops.cast(L_prev_vec, mstype.int32)
    rng_k = ops.Range()(Tensor(0, mstype.int32), kv_len, Tensor(1, mstype.int32))
    rng_r = ops.Range()(Tensor(0, mstype.int32), Tensor(S_cur, mstype.int32), Tensor(1, mstype.int32))
    k_idx = ops.expand_dims(ops.expand_dims(ops.expand_dims(rng_k, 0), 0), 0)   # (1,1,1,kv_len)
    row   = ops.expand_dims(ops.expand_dims(ops.expand_dims(rng_r, 0), 0), -1)  # (1,1,S_cur,1)
    Lp    = ops.expand_dims(ops.expand_dims(ops.expand_dims(L_prev_vec, 1), 1), 1)  # (B,1,1,1)
    allow = ops.less_equal(k_idx, Lp + row)                                      # -> (B,1,S_cur,kv_len)
    return ops.cast(ops.logical_not(allow), mstype.uint8)                         # 0=keep,1=mask

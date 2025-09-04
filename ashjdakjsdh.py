# ---- SAFE GATHER FROM EXTERNAL CACHES (replace your current gather block) ----
kc = key_cache  # external caches from upstream
vc = value_cache
B  = int(block_tables.shape[0])
M  = int(block_tables.shape[1])
bs = int(kc.shape[1])            # block_size
Hk = int(kc.shape[-1])

# Compute how many blocks are valid per batch: m_i = ceil(L_i / bs)
# batch_valid_length: shape [B], int32
L = batch_valid_length
# ceil_div for ints: (x + bs - 1) // bs
m_per_batch = ops.floor_div(ops.add(L, Tensor(bs - 1, L.dtype)), Tensor(bs, L.dtype))  # [B]

# Build a per-batch mask over the block dimension [B, M] telling which entries are valid
# idx = [0,1,2,...,M-1]
idx = ops.arange(0, M, 1, mstype.int32)                      # [M]
idx = ops.expand_dims(idx, 0)                                # [1, M]
m_exp = ops.expand_dims(m_per_batch, 1)                      # [B, 1]
valid_mask = ops.lt(idx, m_exp)                              # [B, M] bool

# Clamp invalid entries to a safe index (0) before gather, then we'll ignore them in attention.
safe_bt = ops.where(valid_mask, block_tables, Tensor(0, mstype.int32))  # [B, M]

# Now gather only with safe indices
flat = ops.reshape(safe_bt, (B * M,))                        # [B*M]
kv_cap = int(ops.reduce_sum(m_per_batch).asnumpy()) * bs if False else M * bs  # only used for comments; not needed

k_full_raw = ops.gather(kc, flat, 0)                         # [B*M, bs, Hk]
v_full_raw = ops.gather(vc, flat, 0)                         # [B*M, bs, Hk]

# Reshape back to [B, M*bs, Hk]; the extra (invalid) tails correspond to masked blocks
k_full = ops.reshape(k_full_raw, (B, M * bs, Hk))            # [B, KV_CAP_PAD, Hk]
v_full = ops.reshape(v_full_raw, (B, M * bs, Hk))

# Optional: if your FA tiler hates long padded K, you can truncate to the max valid KV across batch:
max_m = int(ops.reduce_max(m_per_batch).asnumpy())           # python int
KV = max_m * bs                                              # truncate padded tail
k_full = k_full[:, :KV, :]
v_full = v_full[:, :KV, :]
# ---- END SAFE GATHER ----

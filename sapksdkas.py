# m_per_batch = ceil_div(batch_valid_length, block_size)
L  = batch_valid_length                                # [B], int32
bs = int(key_cache.shape[1])                           # block size (python int)

m_per_batch = ops.floor_div(
    ops.add(L, Tensor(bs - 1, L.dtype)),              # L + (bs-1)
    Tensor(bs, L.dtype)                                # // bs
)                                                      # [B], int32

# build [B, M] valid mask over block tables
idx = ops.arange(0, int(block_tables.shape[1]), 1)     # [M], int64 by default
idx = ops.cast(idx, L.dtype)                           # match L’s dtype (int32)
idx = ops.expand_dims(idx, 0)                          # [1, M]
m_exp = ops.expand_dims(m_per_batch, 1)                # [B, 1]
valid_mask = ops.lt(idx, m_exp)                        # [B, M] bool

# clamp invalid block ids to 0 before gather
safe_bt = ops.where(valid_mask, block_tables, Tensor(0, block_tables.dtype))  # [B, M]

# gather from EXTERNAL caches
B  = int(block_tables.shape[0])
M  = int(block_tables.shape[1])
Hk = int(key_cache.shape[-1])
flat = ops.reshape(safe_bt, (B * M,))                  # [B*M]
k_full = ops.reshape(ops.gather(key_cache, flat, 0), (B, M * bs, Hk))
v_full = ops.reshape(ops.gather(value_cache, flat, 0), (B, M * bs, Hk))

# truncate to max valid KV to avoid padded tail issues with FA tiler
max_m = int(ops.reduce_max(m_per_batch).asnumpy())
KV    = max_m * bs
k_full = k_full[:, :KV, :]
v_full = v_full[:, :KV, :]

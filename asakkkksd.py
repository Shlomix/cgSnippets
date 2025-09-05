from mindspore import ops, Tensor
from mindspore import dtype as mstype

def construct(self, key, value, slot_mapping, _, key_cache=None, value_cache=None):
    """Forward compute of KVCache write for Paged Attention."""

    # Decide which caches we write into (external vs internal)
    use_external = (self.npu_mem_size == -1)
    tgt_kc = key_cache if use_external else self.key_cache
    tgt_vc = value_cache if use_external else self.value_cache

    # ---- UNCONDITIONAL dtype normalization (compile-time safe) ----
    # cache dtype for payloads (BF16/FP16)
    cache_dtype = None
    if tgt_kc is not None and hasattr(tgt_kc, "dtype"):
        cache_dtype = tgt_kc.dtype
    elif tgt_vc is not None and hasattr(tgt_vc, "dtype"):
        cache_dtype = tgt_vc.dtype

    # 1) Make payloads (key/value) EXACTLY the cache dtype
    if cache_dtype is not None and key.dtype != cache_dtype:
        key = ops.cast(key, cache_dtype)
    if cache_dtype is not None and value.dtype != cache_dtype:
        value = ops.cast(value, cache_dtype)

    # 2) Indices must be int32
    if slot_mapping is not None and slot_mapping.dtype != mstype.int32:
        slot_mapping = ops.cast(slot_mapping, mstype.int32)

    # ---- The only kernel that writes Parameters (must see aligned dtypes) ----
    # ReshapeAndCache signature in v1.6: (key, value, key_cache, value_cache, slot_mapping)
    return self.reshape_and_cache(key, value, tgt_kc, tgt_vc, slot_mapping)


from mindspore import ops, Tensor, dtype as mstype

# key/value → external cache dtype (if provided)
if (key_cache is not None) and hasattr(key_cache, "dtype") and (key.dtype != key_cache.dtype):
    key = ops.cast(key, key_cache.dtype)
if (value_cache is not None) and hasattr(value_cache, "dtype") and (value.dtype != value_cache.dtype):
    value = ops.cast(value, value_cache.dtype)

# indices → int32
if slot_mapping.dtype != mstype.int32:
    slot_mapping = ops.cast(slot_mapping, mstype.int32)

# write the chunk
cache_write_out = self.paged_attention_mgr(
    key, value, slot_mapping, batch_valid_length,
    key_cache=key_cache, value_cache=value_cache
)
query = ops.depend(query, cache_write_out)




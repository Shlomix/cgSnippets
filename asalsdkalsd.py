# ========= inside class ParallelPagedAttentionMgr(nn.Cell) =========
from mindspore import ops, Tensor
from mindspore import dtype as mstype

def _cache_dtype(self, key_cache, value_cache):
    # prefer external caches; fall back to internals
    if key_cache is not None and hasattr(key_cache, "dtype"):
        return key_cache.dtype
    if value_cache is not None and hasattr(value_cache, "dtype"):
        return value_cache.dtype
    if getattr(self, "key_cache", None) is not None:
        return self.key_cache.dtype
    if getattr(self, "value_cache", None) is not None:
        return self.value_cache.dtype
    return None

def _normalize_for_cache(self, key, value, slot_mapping, key_cache, value_cache):
    # Make inputs safe for ReshapeAndCache: payloads match cache dtype; indices int32
    cdt = self._cache_dtype(key_cache, value_cache)

    if (cdt is not None) and (key.dtype != cdt):
        key = ops.cast(key, cdt)
    if (cdt is not None) and (value.dtype != cdt):
        value = ops.cast(value, cdt)

    if (slot_mapping is not None) and (slot_mapping.dtype != mstype.int32):
        slot_mapping = ops.cast(slot_mapping, mstype.int32)

    # JIT-safe breadcrumb (ints only). Uncomment for one run:
    # print(88331,
    #       int(cdt is not None),
    #       int(key.dtype == cdt),
    #       int(value.dtype == cdt),
    #       int(slot_mapping is None or slot_mapping.dtype == mstype.int32))
    return key, value, slot_mapping

def construct(self, key, value, slot_mapping, _, key_cache=None, value_cache=None):
    """Forward compute of KVCache write for Paged Attention."""
    key, value, slot_mapping = self._normalize_for_cache(
        key, value, slot_mapping, key_cache, value_cache
    )

    # External-cache mode
    if self.npu_mem_size == -1:
        # v1.6 signature: (key, value, key_cache, value_cache, slot_mapping)
        return self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    # Internal-cache mode
    return self.reshape_and_cache(key, value, self.key_cache, self.value_cache, slot_mapping)
# ========= end patch =========



from mindspore import ops, Tensor, dtype as mstype

# Ensure payloads match external cache dtype (and indices are int32)
if (key_cache is not None) and (hasattr(key_cache, "dtype")) and (key.dtype != key_cache.dtype):
    key = ops.cast(key, key_cache.dtype)
if (value_cache is not None) and (hasattr(value_cache, "dtype")) and (value.dtype != value_cache.dtype):
    value = ops.cast(value, value_cache.dtype)
if slot_mapping.dtype != mstype.int32:
    slot_mapping = ops.cast(slot_mapping, mstype.int32)

# Optional one-run breadcrumb (ints only):
# print(88321, int(key.dtype == key_cache.dtype if key_cache is not None else 1),
#              int(value.dtype == value_cache.dtype if value_cache is not None else 1),
#              int(slot_mapping.dtype == mstype.int32))

# Now write to cache (external mode)
cache_write_out = self.paged_attention_mgr(
    key, value, slot_mapping, batch_valid_length,
    key_cache=key_cache, value_cache=value_cache
)
query = ops.depend(query, cache_write_out)


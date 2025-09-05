# ========= paste this inside class ParallelPagedAttentionMgr(nn.Cell) =========

def _normalize_for_cache(self,
                         key,
                         value,
                         slot_mapping,
                         key_cache=None,
                         value_cache=None):
    """
    Make inputs safe for the auto-generated ReshapeAndCache op:
      - key/value match the cache dtype (BF16/FP16)
      - slot_mapping is int32
    Returns: (key_n, value_n, slot_mapping_n)
    """
    from mindspore import ops, Tensor
    from mindspore import dtype as mstype

    # Determine the cache dtype we will write into (external first, then internal)
    cache_dtype = None
    if key_cache is not None and hasattr(key_cache, "dtype"):
        cache_dtype = key_cache.dtype
    elif value_cache is not None and hasattr(value_cache, "dtype"):
        cache_dtype = value_cache.dtype
    elif getattr(self, "key_cache", None) is not None:
        cache_dtype = self.key_cache.dtype
    elif getattr(self, "value_cache", None) is not None:
        cache_dtype = self.value_cache.dtype

    # 1) Cast payloads to cache dtype (required for in-place writes to Parameter)
    if cache_dtype is not None and key.dtype != cache_dtype:
        key = ops.cast(key, cache_dtype)
    if cache_dtype is not None and value.dtype != cache_dtype:
        value = ops.cast(value, cache_dtype)

    # 2) Routing tensor must be int32
    if slot_mapping is not None and slot_mapping.dtype != mstype.int32:
        slot_mapping = ops.cast(slot_mapping, mstype.int32)

    # (No prints; graph-safe)
    return key, value, slot_mapping


def construct(self, key, value, slot_mapping, _, key_cache=None, value_cache=None):
    """The forward compute of KVCache for Paged Attention."""
    # Normalize dtypes before touching the cache
    key, value, slot_mapping = self._normalize_for_cache(
        key, value, slot_mapping, key_cache=key_cache, value_cache=value_cache
    )

    # External-cache mode: npu_mem_size == -1 -> use caches provided by caller
    if self.npu_mem_size == -1:
        return self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    # Internal-cache mode: manager owns self.key_cache / self.value_cache
    return self.reshape_and_cache(key, value, self.key_cache, self.value_cache, slot_mapping)
# ========= end of patch =========

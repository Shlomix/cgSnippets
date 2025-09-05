# ---- in research/qwen2_5/infer/parallel_paged_attention_mgr.py ----
# inside class ParallelPagedAttentionMgr(nn.Cell)

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
    from mindspore import ops, Tensor
    from mindspore import dtype as mstype

    cdt = self._cache_dtype(key_cache, value_cache)

    # 1) payloads must match cache dtype (BF16/FP16)
    if cdt is not None and key.dtype != cdt:
        key = ops.cast(key, cdt)
    if cdt is not None and value.dtype != cdt:
        value = ops.cast(value, cdt)

    # 2) routing must be int32 (avoid hidden float/int64 promotion)
    if slot_mapping is not None and slot_mapping.dtype != mstype.int32:
        slot_mapping = ops.cast(slot_mapping, mstype.int32)

    # (optional, JIT-safe breadcrumb)
    # print(88331, int(cdt is not None), int(key.dtype == cdt), int(value.dtype == cdt),
    #               int(slot_mapping is None or slot_mapping.dtype == mstype.int32))
    return key, value, slot_mapping

def construct(self, key, value, slot_mapping, _, key_cache=None, value_cache=None):
    """Forward compute to write K/V into the paged KV cache."""
    # normalize dtypes BEFORE hitting the auto-generated kernel
    key, value, slot_mapping = self._normalize_for_cache(
        key, value, slot_mapping, key_cache, value_cache
    )

    # External-cache mode (npu_mem_size == -1): write into caller-provided buffers
    if self.npu_mem_size == -1:
        # ReshapeAndCache signature in v1.6 is (key, value, key_cache, value_cache, slot_mapping)
        return self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    # Internal-cache mode: write into self.key_cache/self.value_cache
    return self.reshape_and_cache(key, value, self.key_cache, self.value_cache, slot_mapping)

# parallel_paged_attention_mgr.py

from mindspore import ops, Tensor
from mindspore import dtype as mstype

def reshape_and_cache(self,
                      key, value,
                      slot_mapping,            # [B, S] int
                      batch_valid_length,      # [B]    int
                      # ... other args ...,
                      key_cache=None, value_cache=None,
                      ):
    # -------------------- DTYPE SAFETY BLOCK (ADD THIS) --------------------
    # Determine cache dtype (BF16/FP16) from the actual buffers we will write into
    cache_dtype = None
    if key_cache is not None and hasattr(key_cache, "dtype"):
        cache_dtype = key_cache.dtype
    elif value_cache is not None and hasattr(value_cache, "dtype"):
        cache_dtype = value_cache.dtype
    else:
        # Internal mode: use the manager's internal buffers
        if getattr(self, "key_cache", None) is not None:
            cache_dtype = self.key_cache.dtype
        elif getattr(self, "value_cache", None) is not None:
            cache_dtype = self.value_cache.dtype

    # 1) Cast K/V to the cache dtype (required for in-place writes into Parameter)
    if (cache_dtype is not None) and (key.dtype != cache_dtype):
        key = ops.cast(key, cache_dtype)
    if (cache_dtype is not None) and (value.dtype != cache_dtype):
        value = ops.cast(value, cache_dtype)

    # 2) Routing/index tensors must be int32 for scatter/gather kernels
    if slot_mapping is not None and slot_mapping.dtype != mstype.int32:
        slot_mapping = ops.cast(slot_mapping, mstype.int32)
    if batch_valid_length is not None and batch_valid_length.dtype != mstype.int32:
        batch_valid_length = ops.cast(batch_valid_length, mstype.int32)

    # If this function also receives block_tables / block_idx / offsets, cast them too:
    if hasattr(self, "block_tables") and self.block_tables is not None and self.block_tables.dtype != mstype.int32:
        self.block_tables = ops.cast(self.block_tables, mstype.int32)
    # If arguments exist:
    # if block_tables is not None and block_tables.dtype != mstype.int32:
    #     block_tables = ops.cast(block_tables, mstype.int32)

    # 3) Replace bare Python scalars that might float-promote to float64
    #    Use typed Tensor scalars instead whenever they feed ops that touch the cache.
    zero_i32 = Tensor(0, mstype.int32)
    one_i32  = Tensor(1, mstype.int32)
    zero_f   = Tensor(0.0, cache_dtype) if cache_dtype is not None else None
    one_f    = Tensor(1.0, cache_dtype) if cache_dtype is not None else None
    # ----------------------------------------------------------------------
    # ... proceed with your existing reshape + scatter/write into key_cache/value_cache ...

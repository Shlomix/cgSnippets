# --- make inputs dtype-compatible with in-place cache write ---
if not self.use_multi_latent_attention:
    # print dtypes (helps diagnose odd caches like float64)
    if self.debug:
        self._Print("reshape_and_cache dtypes:",
                    " key:", str(key.dtype), " value:", str(value.dtype),
                    " key_cache:", str(key_cache.dtype) if key_cache is not None else "None",
                    " value_cache:", str(value_cache.dtype) if value_cache is not None else "None")

    # key/value must match cache dtype; cast the *inputs* (Parameters cannot be cast)
    if key_cache is not None and key is not None and key.dtype != key_cache.dtype:
        self._dbg("reshape_and_cache: casting key from ", str(key.dtype), " to ", str(key_cache.dtype))
        key = self._Cast(key, key_cache.dtype)

    if value_cache is not None and value is not None and value.dtype != value_cache.dtype:
        self._dbg("reshape_and_cache: casting value from ", str(value.dtype), " to ", str(value_cache.dtype))
        value = self._Cast(value, value_cache.dtype)

    # slot_mapping must be int32
    if slot_mapping is not None and slot_mapping.dtype != mstype.int32:
        self._dbg("reshape_and_cache: casting slot_mapping from ", str(slot_mapping.dtype), " to int32")
        slot_mapping = self._Cast(slot_mapping, mstype.int32)

    # now it’s safe to write into the caches
    self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

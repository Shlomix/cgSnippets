def _ensure_tnd_causal_mask(self, attn_mask, size=2048):
    """
    TND requires a (2048, 2048) lower-triangular mask with 1=discard, 0=keep.
    If incoming mask doesn't match, build the canonical one.
    """
    if self._tnd_mask_2048 is None:
        ones_bool = ops.Ones()( (size, size), mstype.bool_ )
        upper_bool = self._triu_strict(ones_bool)          # diag=1 baked in
        self._tnd_mask_2048 = self._cast(upper_bool, mstype.uint8)

    if attn_mask is not None:
        if len(attn_mask.shape) == 2 and attn_mask.shape[0] == size and attn_mask.shape[1] == size:
            if attn_mask.dtype == mstype.uint8 or attn_mask.dtype == mstype.bool_:
                return attn_mask
    return self._tnd_mask_2048


def _kv_from_cache_tnd(self, cache, block_tables, actual_seq_kvlen):
    """
    Build (T2, N_kv, D) from block-wise cache using block_tables and ragged kv lengths.
    cache: (num_blocks, block_size, N_kv, D)
    block_tables: (B, max_blocks_per_seq) int32
    actual_seq_kvlen: cumulative kv lengths [B], last == T2 (may be Parameter)
    """
    nb, bs, n_kv, d = cache.shape
    flat = self._reshape(cache, (nb * bs, n_kv, d))  # (nb*bs, N_kv, D)

    # lengths per sequence (keep original dtype; do NOT cast Parameters)
    kv_cum = actual_seq_kvlen
    kv_lens = self._diff_lengths(kv_cum)             # [B], same dtype as input
    B = kv_lens.shape[0]

    # Build positions [0..max_len-1] as int32 (safe synthetic tensor)
    max_len = self._reduce_max(kv_lens)              # scalar tensor
    if max_len.dtype != mstype.int32:
        max_len_i32 = self._cast(max_len, mstype.int32)
    else:
        max_len_i32 = max_len
    pos_i32 = self._range(self._cast(0, mstype.int32),
                          max_len_i32,
                          self._cast(1, mstype.int32))           # (L,)

    # Validity mask uses kv_lens dtype; cast only the synthetic pos
    pos_for_mask = self._cast(self._expand_dims(pos_i32, 0), kv_lens.dtype)  # (1, L) -> dtype of kv_lens
    kv_lens_2d = self._expand_dims(kv_lens, 1)                               # (B, 1)
    valid_mask = self._greater(kv_lens_2d, pos_for_mask)                     # (B, L)  True where pos < len

    # Compute block indices without Tile by broadcasting with zeros of shape (B,1)
    zeros_B1 = self._zeros((B, 1), mstype.int32)
    blk_idx_row = pos_i32 // bs                                             # (L,)
    blk_idx = zeros_B1 + blk_idx_row                                        # (B,1)+(L,) -> (B,L) via broadcast

    # Gather block ids and compute global indices (all int32)
    table_i32 = self._cast(block_tables, mstype.int32)                      # (B, max_blocks)
    blk_ids = self._gather_d(table_i32, 1, blk_idx)                         # (B, L)
    offsets_row = pos_i32 - (blk_idx_row * bs)                              # (L,)
    offsets = zeros_B1 + offsets_row                                        # (B, L)
    global_idx = blk_ids * bs + offsets                                     # (B, L)

    # Ragged pack: select valid positions to get token-major indices [T2]
    valid_idx_flat = self._masked_select(global_idx, valid_mask)            # (T2,)
    kv_tnd = self._gather(flat, valid_idx_flat, 0)                          # (T2, N_kv, D)
    return kv_tnd

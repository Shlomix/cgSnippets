self._triu = ops.Triu()
self._ones = ops.Ones()


def _ensure_tnd_causal_mask(self, attn_mask, size=2048):
    """
    TND requires a lower-triangular mask of shape (2048, 2048),
    dtype bool/uint8, with 1=discard and 0=keep. If incoming mask doesn't
    match, build the canonical one.
    """
    if self._tnd_mask_2048 is None:
        # Build strictly upper-triangular ones (discard=1) with zeros elsewhere.
        # This avoids Tile entirely (dims must be tuple[int] in MindSpore).
        ones = self._ones((size, size), mstype.uint8)   # constant shape
        upper = self._triu(ones, 1)                     # strictly upper triangle = 1
        self._tnd_mask_2048 = upper                     # 1=discard, 0=keep

    if attn_mask is not None:
        if len(attn_mask.shape) == 2 and attn_mask.shape[0] == size and attn_mask.shape[1] == size:
            if attn_mask.dtype == mstype.uint8 or attn_mask.dtype == mstype.bool_:
                return attn_mask
    return self._tnd_mask_2048

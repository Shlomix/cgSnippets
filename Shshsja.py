# --- put this inside your attention class, replacing _bsh_to_bnsd ---

from mindspore import ops

def _to_bnsd_or_none(self, x, n_heads: int):
    """
    Convert x to [B,N,S,D] for FlashAttention(input_layout='BNSD').
    Accepts TH [T,H], BSH [B,S,H], or already BNSD [B,N,S,D].
    Returns a Tensor or None (None => can't convert cleanly, e.g. GQA H % n_heads != 0).
    """
    shp = getattr(x, "shape", ())
    r = len(shp)
    if r == 4:
        # already BNSD
        return x
    if r == 3:
        B, S, H = int(shp[0] or 1), int(shp[1] or 1), int(shp[2] or 1)
        if n_heads <= 0 or (H % n_heads) != 0:
            return None
        D = H // n_heads
        # B,S,H -> B,S,N,D -> B,N,S,D
        return ops.transpose(ops.reshape(x, (B, S, n_heads, D)), (0, 2, 1, 3))
    if r == 2:
        # TH -> assume B=1
        T, H = int(shp[0] or 1), int(shp[1] or 1)
        if n_heads <= 0 or (H % n_heads) != 0:
            return None
        D = H // n_heads
        # 1,T,H -> 1,T,N,D -> 1,N,T,D (S := T)
        return ops.transpose(ops.reshape(x, (1, T, n_heads, D)), (0, 2, 1, 3))
    # unsupported rank
    return None

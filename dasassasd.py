# =====================================
# 1) ADD THIS IMPORT AT THE TOP (near other imports)
# =====================================
import os


# =====================================
# 2) __init__ additions
#    Put these lines INSIDE InferAttention.__init__(...), AFTER input_layout/use_attention_mask
#    are set, and BEFORE constructing self.flash_attention.
# =====================================
self.debug = (os.getenv("MF_DEBUG", "0") == "1")

# Prefer causal FA (no dense mask) for chunk prefill or parallel-style prefill.
# This ensures we can drive FA using lengths only.
if (self.use_flash_attention and (parallel_decoding or chunk_prefill)):
    self.sparse_mode = 2            # causal (left-up)
    self.use_attention_mask = False # rely on causal kernel, not a dense mask
    if self.debug:
        print("[InferAttention.__init__] enable causal FA for 2D/chunk prefill "
              f"(sparse_mode={self.sparse_mode}, use_attention_mask={self.use_attention_mask}, "
              f"input_layout={self.input_layout})")


# =====================================
# 3) REPLACE _prefill_attention WITH THIS VERSION
#    (same name/signature; only adds passing of ragged lengths + prints)
# =====================================
def _prefill_attention(self, query, key, value, attn_mask, alibi_mask,
                       actual_seq_qlen=None, actual_seq_kvlen=None):
    """
    prefill attention:
      - If FlashAttention is enabled, use it.
      - If lengths are provided, pass them to FA so it can run causal without a dense mask.
    """
    if self.debug:
        print("[InferAttention.prefill] use_FA=", self.use_flash_attention,
              " layout=", self.input_layout)
        if actual_seq_qlen is not None:
            print("[InferAttention.prefill]   actual_seq_qlen=", actual_seq_qlen)
        if actual_seq_kvlen is not None:
            print("[InferAttention.prefill]   actual_seq_kvlen=", actual_seq_kvlen)
        if attn_mask is not None:
            print("[InferAttention.prefill]   attn_mask provided")

    if self.input_layout == "TH":
        if self.use_flash_attention:
            # query shape: [1, S, H] -> [S, H]
            bs, seq_len, _ = query.shape
            query = self.reshape(query, (-1, self.n_head * self.head_dim))
            key = self.reshape(key, (-1, self.n_kv_head * self.head_dim))
            value = self.reshape(value, (-1, self.n_kv_head * self.head_dim))
            # FA call; pass lengths if provided
            out = self.flash_attention(query, key, value, attn_mask, alibi_mask,
                                       None, None,  # padding_mask, prefix (unused here)
                                       actual_seq_qlen, actual_seq_kvlen)
            out = self.reshape(out, (bs, seq_len, self.n_head * self.head_dim))
            if self.debug:
                print("[InferAttention.prefill]   FA(TH) done")
            return out
        # fallback (non-FA TH not supported here)
        return self._core_attention_th(query, key, value, attn_mask, alibi_mask)

    if self.input_layout == "BSH":
        if self.use_flash_attention:
            # In BSH we also pass lengths if available; when using causal FA (sparse_mode=2),
            # attn_mask is typically None.
            if (actual_seq_qlen is not None) and (actual_seq_kvlen is not None):
                out = self.flash_attention(query, key, value, attn_mask, alibi_mask,
                                           None, None,  # padding_mask, prefix
                                           actual_seq_qlen, actual_seq_kvlen)
            else:
                out = self.flash_attention(query, key, value, attn_mask, alibi_mask)
            if self.debug:
                print("[InferAttention.prefill]   FA(BSH) done")
            return out
        # fallback (no-FA) uses your softmax path with BSH reshape
        return self._core_attention_bsh(query, key, value, attn_mask, alibi_mask)

    raise ValueError(f"FlashAttention input layout:{self.input_layout} is not supported.")


# =====================================
# 4) construct() edits
#    Replace ONLY the body after writing KV, i.e., the part that decides chunk prefill / prefill / decode.
#    Keep your KV write call exactly as-is.
# =====================================

# KV write (unchanged)
key_out = self.paged_attention_mgr(key, value, slot_mapping, batch_valid_length)
query = ops.depend(query, key_out)

# --- Debug header for this iteration ---
if self.debug:
    mode = "chunk_prefill" if self.chunk_prefill else ("prefill" if self.is_first_iteration else "decode")
    print(f"[InferAttention.construct] mode={mode}")
    if q_seq_lens is not None:
        print("[InferAttention.construct]   q_seq_lens=", q_seq_lens)
    if batch_valid_length is not None:
        print("[InferAttention.construct]   batch_valid_length(prefix)=", batch_valid_length)

# 4.a) CHUNK PREFILL BRANCH
if self.chunk_prefill:
    # If FA is enabled and we have 2D chunk info, use FA from 2nd chunk onward.
    use_fa = (self.use_flash_attention and (q_seq_lens is not None))
    # consider "second chunk onward" as prefix>0; first chunk keeps paged for maximum safety
    prefix_positive = False
    try:
        # batch_valid_length is int32 Tensor; ReduceMax works in both modes
        prefix_positive = (ops.ReduceMax(keep_dims=False)(batch_valid_length) > 0)
    except Exception:
        pass

    if use_fa and prefix_positive:
        # causal FA: no dense mask
        attn_mask_for_fa = None if self.sparse_mode == 2 else attn_mask
        actual_q = q_seq_lens
        actual_kv = batch_valid_length + q_seq_lens
        if self.debug:
            print("[InferAttention.construct]   CHUNK->FA  actual_q=", actual_q,
                  " actual_kv=", actual_kv, " (prefix+chunk)")
        return self._prefill_attention(query, key, value, attn_mask_for_fa, alibi_mask,
                                       actual_seq_qlen=actual_q, actual_seq_kvlen=actual_kv)

    # otherwise fall back to paged attention for chunk prefill (original behavior)
    if self.debug:
        why = "first_chunk" if use_fa and not prefix_positive else "no_FA_or_no_q_seq_lens"
        print(f"[InferAttention.construct]   CHUNK->Paged ({why})")
    return self.paged_attention_mgr.paged_attn(query, batch_valid_length, block_tables,
                                               attn_mask=attn_mask, q_seq_lens=q_seq_lens)

# 4.b) NON-CHUNK PREFILL (first pass)
if self.is_first_iteration:
    # If q_seq_lens is provided even in non-chunk mode, honor it; else use batch_valid_length.
    actual_q = q_seq_lens if q_seq_lens is not None else batch_valid_length
    # kv len: if q_seq_lens is None (plain prefill), kv == q == batch_valid_length
    actual_kv = (batch_valid_length + q_seq_lens) if q_seq_lens is not None else batch_valid_length
    if self.debug:
        print("[InferAttention.construct]   PREFILL->FA  actual_q=", actual_q,
              " actual_kv=", actual_kv)
    return self._prefill_attention(query, key, value, attn_mask, alibi_mask,
                                   actual_seq_qlen=actual_q, actual_seq_kvlen=actual_kv)

# 4.c) DECODE (unchanged)
return self._incre_attention(query, batch_valid_length, block_tables, alibi_mask, attn_mask, q_seq_lens)

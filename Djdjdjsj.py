# ========= [1] add near top imports =========
import os


# ========= [2] inside class Attention.__init__(...) after existing fields are set (e.g., after self.linear_proj) =========
# env toggles
self._force_fa_for_chunk      = (os.getenv("MF_FORCE_FA_CHUNK", "0") == "1")
self._disable_chunk_parallel  = (os.getenv("MF_DISABLE_CHUNK_PARALLEL", "0") == "1")
# when set, we *intentionally* pass KV length that excludes the current chunk (prefix only) to provoke FA failure
self._fa_prefix_only          = (os.getenv("MF_FA_PREFIX_ONLY", "0") == "1")

# graph-safe printer
self._pp = ops.Print()


# ========= [3] in Attention.construct(...), just BEFORE calling self.core_attention(...) =========
# Make local copies so we don’t mutate the original args.
_q_seq_lens = q_seq_lens
_attn_mask  = attention_mask
_act_qlen   = actual_seq_qlen
_act_kvlen  = actual_seq_kvlen

# --- (A) Optional: disable the 2D/chunk path entirely at this layer ---
if self._disable_chunk_parallel and _q_seq_lens is not None:
    self._pp("ATTN: MF_DISABLE_CHUNK_PARALLEL=1 → ignoring q_seq_lens at this layer.")
    _q_seq_lens = None

# --- (B) Always guarantee decode sees a valid q_seq_lens vector when the core uses parallel decode.
# If upstream didn’t provide one, synthesize `[1]*batch` so the PagedAttention kernel can tile.
if _q_seq_lens is None and batch_valid_length is not None:
    _q_seq_lens_for_call = mint.full_like(batch_valid_length, 1)      # Int32, shape [B]
else:
    _q_seq_lens_for_call = _q_seq_lens

# --- (C) If we want FlashAttention for 2D chunk prefill, prepare ragged lengths and drop dense mask.
_use_fa_chunk = bool(self._force_fa_for_chunk and self.use_flash_attention and (_q_seq_lens is not None))
if _use_fa_chunk:
    # causal FA: no dense attn_mask
    _attn_mask = None
    # actual query length is the current chunk lengths
    if _act_qlen is None:
        _act_qlen = _q_seq_lens
    # actual KV length: normally prefix + chunk; to *provoke failure*, pass prefix only when MF_FA_PREFIX_ONLY=1
    if _act_kvlen is None:
        if self._fa_prefix_only:
            _act_kvlen = batch_valid_length                      # <— KV shorter than Q (will crash FA)
        else:
            _act_kvlen = batch_valid_length + _q_seq_lens        # <— normal full-context prefill
    self._pp("ATTN: FA for 2D chunk | act_q=", _act_qlen, " act_kv=", _act_kvlen)

# ========= [4] now call core_attention with our prepared locals =========
if self.use_flash_attention:
    core_attn_out = self.core_attention(
        query=query,
        key=key,
        value=value,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        batch_valid_length=batch_valid_length,
        context_lens_tensor=context_lens_tensor,
        q_seq_lens=_q_seq_lens_for_call,   # always valid for paged decode (>= ones)
        actual_seq_qlen=_act_qlen,         # used by FA when we’re in 2D chunk mode
        actual_seq_kvlen=_act_kvlen,       # used by FA when we’re in 2D chunk mode
        attn_mask=_attn_mask,              # None for FA chunk (causal), original otherwise
        key_cache=key_cache,
        value_cache=value_cache)
else:
    core_attn_out = self.core_attention(query, key, value, attention_mask)

# =========================
# [A] ADD NEAR THE TOP IMPORTS
# (alongside the existing imports in this file)
# =========================
import os  # NEW


# =========================
# [B] INSIDE class Attention.__init__(...) AFTER existing fields are set
# (e.g., after self.linear_proj is built is fine, or right after self.core_attention)
# =========================
# --- runtime knobs ---
# Use FA for 2D/chunk prefill when q_seq_lens is present
self._force_fa_for_chunk = (os.getenv("MF_FORCE_FA_CHUNK", "0") == "1")
# Disable the 2D/chunk path locally by nulling q_seq_lens before core_attention
self._disable_chunk_parallel = (os.getenv("MF_DISABLE_CHUNK_PARALLEL", "0") == "1")

# graph-safe logger (optional, cheap)
self._pp = ops.Print()


# =========================
# [C] IN Attention.construct(...), just BEFORE the "if self.use_flash_attention:" call,
#     insert this small "prep" block that adjusts what we pass through.
#     (We DO NOT change the function signature; we only alter local variables passed to core_attention.)
# =========================
# --- begin FA/paged prep for 2D chunk handling ---
# Work on local copies so we don't disturb original args.
_q_seq_lens = q_seq_lens
_attn_mask  = attention_mask
_act_qlen   = actual_seq_qlen
_act_kvlen  = actual_seq_kvlen

# 1) Optionally disable the 2D/chunk path entirely at this layer
if self._disable_chunk_parallel and _q_seq_lens is not None:
    # By dropping q_seq_lens here, the “chunk/parallel” path in the core will not trigger.
    self._pp("ATTN: MF_DISABLE_CHUNK_PARALLEL=1 → ignoring q_seq_lens.")
    _q_seq_lens = None
    # Leave other args as-is; decode still uses paged/cache as usual.

# 2) If we *want* FA for chunk prefill and we’re using FA at this layer,
#    pass ragged lengths so FA can run causal without a dense mask.
_use_fa_for_chunk_now = bool(self._force_fa_for_chunk and self.use_flash_attention and (_q_seq_lens is not None))
if _use_fa_for_chunk_now:
    # Causal FA: no dense mask required.
    _attn_mask = None
    # If caller didn't precompute actual_seq_* for FA, derive them here:
    if _act_qlen is None:
        _act_qlen = _q_seq_lens
    if _act_kvlen is None:
        # Typical semantics: kv length = prefix length tracked upstream.
        # If upstream passes batch_valid_length as “prefix+this_chunk”, keep it as-is;
        # otherwise you can add q_seq_lens on the producer side. Here we pass what we have.
        _act_kvlen = batch_valid_length
    self._pp("ATTN: MF_FORCE_FA_CHUNK=1 → FlashAttention for 2D chunk.",
             " q_seq_lens=", _q_seq_lens, " act_qlen=", _act_qlen, " act_kvlen=", _act_kvlen)
# --- end FA/paged prep for 2D chunk handling ---


# =========================
# [D] NOW REPLACE JUST THE "if self.use_flash_attention:" CALL SITE
#     to pass our local variables (_attn_mask, _q_seq_lens, _act_qlen, _act_kvlen)
# =========================
if self.use_flash_attention:
    core_attn_out = self.core_attention(
        query=query,
        key=key,
        value=value,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        batch_valid_length=batch_valid_length,
        context_lens_tensor=context_lens_tensor,
        q_seq_lens=_q_seq_lens,             # <— possibly None if disabled
        actual_seq_qlen=_act_qlen,          # <— provided for FA ragged prefill
        actual_seq_kvlen=_act_kvlen,        # <— provided for FA ragged prefill
        attn_mask=_attn_mask,               # <— None for causal FA chunk
        key_cache=key_cache,
        value_cache=value_cache)
else:
    core_attn_out = self.core_attention(query, key, value, attention_mask)

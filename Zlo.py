# ============================================================
# [1] ADD NEAR THE TOP IMPORTS (alongside existing imports)
# ============================================================
import os  # NEW


# ============================================================
# [2] INSIDE class Attention.__init__(...) AFTER your existing fields are set
#     (right after self.linear_proj is built is a fine spot)
# ============================================================
# Graph-safe printer (optional breadcrumbs during runtime; safe in graph mode)
self._pp = ops.Print()

# Nothing else needed in __init__ — we’ll keep this patch strictly local to construct().


# ============================================================
# [3] IN Attention.construct(...), INSERT THIS PREP/CALL BLOCK
#     JUST BEFORE the existing `if self.use_flash_attention:` that calls self.core_attention(...).
#     (We do not change the function signature; we only prepare the arguments.)
# ============================================================

# ---- A) Sanitize control-tensor dtypes the kernels expect (int32) ----
if batch_valid_length is not None and batch_valid_length.dtype != mint.int32:
    batch_valid_length = mint.astype(batch_valid_length, mint.int32)

if q_seq_lens is not None and q_seq_lens.dtype != mint.int32:
    q_seq_lens = mint.astype(q_seq_lens, mint.int32)

if block_tables is not None and block_tables.dtype != mint.int32:
    block_tables = mint.astype(block_tables, mint.int32)

if slot_mapping is not None and slot_mapping.dtype != mint.int32:
    slot_mapping = mint.astype(slot_mapping, mint.int32)

# ---- B) Always give the paged path a q_seq_lens vector for decode graph build ----
# (Decode typically uses ones; without this, paged tiler can fail during compile.)
if q_seq_lens is None and batch_valid_length is not None:
    q_seq_lens_for_paged = mint.full_like(batch_valid_length, 1)  # int32 [B]
else:
    q_seq_lens_for_paged = q_seq_lens

# ---- C) Decide the path for THIS call:
#   • If q_seq_lens is present → this is a chunk-prefill call (we treat it as second+ chunk)
#     → run FlashAttention with ragged lengths.
#   • Otherwise → original flow (first prefill or decode), using paged as before.
#
# NOTE: This implements exactly “first chunk original; second+ chunk FA” under the common
#       convention that q_seq_lens is only passed from the 2nd chunk onward.
# ---------------------------------------------------------------------
if self.use_flash_attention and (q_seq_lens is not None):
    # === FlashAttention for chunk-prefill (second+ chunk) ===
    # Causal FA → no dense mask.
    attn_mask_for_fa = None

    # Ragged lengths for FA (per-batch vectors):
    #   Q length = this chunk
    actual_seq_qlen = q_seq_lens
    #   KV length = prefix + current chunk
    actual_seq_kvlen = batch_valid_length + q_seq_lens

    # Scrub paged-only knobs so core can’t choose paged here:
    slot_mapping_for_call   = None
    block_tables_for_call   = None
    key_cache_for_call      = None
    value_cache_for_call    = None
    context_lens_for_call   = None

    # Breadcrumb (optional)
    self._pp("ATTN[CHUNK→FA]: using FlashAttention for chunk-prefill.",
             " q=", actual_seq_qlen, " kv=", actual_seq_kvlen)

    # Call the core with FA-oriented args
    core_attn_out = self.core_attention(
        query=query,
        key=key,
        value=value,
        slot_mapping=slot_mapping_for_call,
        block_tables=block_tables_for_call,
        batch_valid_length=batch_valid_length,
        context_lens_tensor=context_lens_for_call,
        q_seq_lens=q_seq_lens,                 # present → this is a chunk call
        actual_seq_qlen=actual_seq_qlen,       # drives FA ragged prefill
        actual_seq_kvlen=actual_seq_kvlen,     # drives FA ragged prefill
        attn_mask=attn_mask_for_fa,            # causal FA
        key_cache=key_cache_for_call,
        value_cache=value_cache_for_call
    )
else:
    # === Original flow for first prefill OR decode ===
    # Keep your existing behavior; just ensure paged sees a valid q_seq_lens for compile.
    self._pp("ATTN[ORIGINAL]: first prefill or decode (paged path stays).")
    if self.use_flash_attention:
        core_attn_out = self.core_attention(
            query=query,
            key=key,
            value=value,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            q_seq_lens=q_seq_lens_for_paged,   # ones for decode if original q_seq_lens is None
            actual_seq_qlen=actual_seq_qlen,   # pass-through if caller set them
            actual_seq_kvlen=actual_seq_kvlen,
            attn_mask=attention_mask,
            key_cache=key_cache,
            value_cache=value_cache
        )
    else:
        core_attn_out = self.core_attention(query, key, value, attention_mask)

# (then your existing projection step continues)
# output = self.linear_proj(core_attn_out)
# output = self.cast(output, ori_dtype)
# return output

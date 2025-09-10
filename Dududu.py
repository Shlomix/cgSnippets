# ============================================================
# [1] ADD NEAR THE TOP IMPORTS (with existing imports)
# ============================================================
import os  # NEW


# ============================================================
# [2] INSIDE class Attention.__init__(...) AFTER existing fields are set
#     (right after self.linear_proj is built is a good spot)
# ============================================================
# Toggle: keep absolutely original flow (no chunk→FA switch) when set to 1
#   export MF_USE_ORIGINAL_FLOW=1
self._use_original_flow = (os.getenv("MF_USE_ORIGINAL_FLOW", "0") == "1")

# Optional graph-safe print (breadcrumbs in graph mode)
self._pp = ops.Print()


# ============================================================
# [3] IN Attention.construct(...), INSERT THIS PREP/CALL BLOCK
#     JUST BEFORE your existing `if self.use_flash_attention:` call to self.core_attention(...)
#     (We do not change the function signature; we only prepare the arguments we pass.)
# ============================================================

# ---- A) Sanitize control-tensor dtypes expected by kernels (int32) ----
if batch_valid_length is not None and batch_valid_length.dtype != mint.int32:
    batch_valid_length = mint.astype(batch_valid_length, mint.int32)

if q_seq_lens is not None and q_seq_lens.dtype != mint.int32:
    q_seq_lens = mint.astype(q_seq_lens, mint.int32)

if block_tables is not None and block_tables.dtype != mint.int32:
    block_tables = mint.astype(block_tables, mint.int32)

if slot_mapping is not None and slot_mapping.dtype != mint.int32:
    slot_mapping = mint.astype(slot_mapping, mint.int32)

# ---- B) Always give PagedAttention a q_seq_lens vector for decode graph build ----
# (Decode uses ones; without this, paged tiler can fail during compile.)
if q_seq_lens is None and batch_valid_length is not None:
    q_seq_lens_for_paged = mint.full_like(batch_valid_length, 1)  # int32 [B]
else:
    q_seq_lens_for_paged = q_seq_lens

# ---- C) Decide path for THIS call -------------------------------------
# Flag ON → original flow, no changes anywhere.
# Flag OFF → if q_seq_lens present (i.e., chunk-prefill second+ chunk),
#            route this call to FlashAttention; otherwise keep original flow.
if self._use_original_flow:
    # === ORIGINAL FLOW (unchanged) =====================================
    # Small marker (optional)
    self._pp("ATTN[ORIGINAL_FLOW]: running unmodified path.")
    if self.use_flash_attention:
        core_attn_out = self.core_attention(
            query=query,
            key=key,
            value=value,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            q_seq_lens=q_seq_lens_for_paged,   # ones for decode if q_seq_lens was None
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            attn_mask=attention_mask,
            key_cache=key_cache,
            value_cache=value_cache
        )
    else:
        core_attn_out = self.core_attention(query, key, value, attention_mask)

else:
    # === NEW BEHAVIOR: chunk-prefill (second+) → FlashAttention =========
    if self.use_flash_attention and (q_seq_lens is not None):
        # We consider presence of q_seq_lens as “second+ chunk” in chunk-prefill.
        # Use FA with ragged lengths; causal FA → no dense mask required.
        attn_mask_for_fa = None
        actual_seq_qlen_new = q_seq_lens
        actual_seq_kvlen_new = batch_valid_length + q_seq_lens

        # Scrub paged-only args so the core cannot choose paged for this call.
        slot_mapping_fa   = None
        block_tables_fa   = None
        key_cache_fa      = None
        value_cache_fa    = None
        context_lens_fa   = None

        # Optional breadcrumb
        self._pp("ATTN[CHUNK→FA]: FlashAttention for chunk-prefill.",
                 " q=", actual_seq_qlen_new, " kv=", actual_seq_kvlen_new)

        core_attn_out = self.core_attention(
            query=query,
            key=key,
            value=value,
            slot_mapping=slot_mapping_fa,
            block_tables=block_tables_fa,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_fa,
            q_seq_lens=q_seq_lens,                   # present → chunk call
            actual_seq_qlen=actual_seq_qlen_new,     # FA ragged lengths
            actual_seq_kvlen=actual_seq_kvlen_new,   # FA ragged lengths
            attn_mask=attn_mask_for_fa,              # causal FA
            key_cache=key_cache_fa,
            value_cache=value_cache_fa
        )
    else:
        # First prefill (no q_seq_lens) or decode → original path
        self._pp("ATTN[ORIGINAL]: first prefill or decode (paged/original).")
        if self.use_flash_attention:
            core_attn_out = self.core_attention(
                query=query,
                key=key,
                value=value,
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                batch_valid_length=batch_valid_length,
                context_lens_tensor=context_lens_tensor,
                q_seq_lens=q_seq_lens_for_paged,     # ones for decode if needed
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                attn_mask=attention_mask,
                key_cache=key_cache,
                value_cache=value_cache
            )
        else:
            core_attn_out = self.core_attention(query, key, value, attention_mask)

# (your file already continues with:)
# output = self.linear_proj(core_attn_out)
# output = self.cast(output, ori_dtype)
# return output

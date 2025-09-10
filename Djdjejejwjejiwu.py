# =========================
# [A] ADD NEAR THE TOP IMPORTS (with the other imports)
# =========================
import os  # NEW


# =========================
# [B] INSIDE class Attention.__init__(...) AFTER your existing fields are set
#     (right after self.linear_proj is built is a good spot)
# =========================
# --- tiny runtime toggles ---
# Set MF_FA_PREFIX_ONLY=1 to *intentionally* make FA see kv_len < q_len (to show the FA limitation).
self._fa_prefix_only = (os.getenv("MF_FA_PREFIX_ONLY", "0") == "1")

# graph-safe printer (shows in graph mode)
self._pp = ops.Print()


# =========================
# [C] IN Attention.construct(...), INSERT THIS "PREP" BLOCK JUST BEFORE the
#     existing `if self.use_flash_attention:` call that invokes `self.core_attention(...)`.
# =========================
# ---- sanitize dtypes (PagedAttention kernel requires int32 inputs) ----
if batch_valid_length is not None and batch_valid_length.dtype != mint.int32:
    batch_valid_length = mint.astype(batch_valid_length, mint.int32)

if q_seq_lens is not None and q_seq_lens.dtype != mint.int32:
    q_seq_lens = mint.astype(q_seq_lens, mint.int32)

if block_tables is not None and block_tables.dtype != mint.int32:
    block_tables = mint.astype(block_tables, mint.int32)

if slot_mapping is not None and slot_mapping.dtype != mint.int32:
    slot_mapping = mint.astype(slot_mapping, mint.int32)

# ---- guarantee PagedAttention tiler always sees a q_seq_lens vector ----
# (In decode, it can be a vector of ones; without this the tiler can fail.)
if q_seq_lens is None and batch_valid_length is not None:
    q_seq_lens = mint.full_like(batch_valid_length, 1)  # int32, shape [B]

# ---- optional: prepare FA ragged lengths for the demo (kv < q) ----
# Only set these when you want to *demonstrate* the FA limitation.
# This does not force FA; it only prepares the args so that when the FA path is used
# (e.g., 2D/chunk prefill), FA will see kv_len < q_len and fail as requested.
if self._fa_prefix_only and q_seq_lens is not None:
    # q_len = current chunk; kv_len = prefix only  → kv_len < q_len (will cause FA to error)
    actual_seq_qlen = q_seq_lens
    actual_seq_kvlen = batch_valid_length
    # small breadcrumb visible in graph logs
    self._pp("ATTN[FA_DEMO]: kv_len < q_len → expecting FlashAttention to reject.",
             " q_seq_lens=", q_seq_lens, " prefix(batch_valid_length)=", batch_valid_length)

# (No other changes below—call self.core_attention exactly as the file already does,
# but with the possibly-updated q_seq_lens / actual_seq_* / block_tables / slot_mapping)
# Example (this is how your file already calls it):
# if self.use_flash_attention:
#     core_attn_out = self.core_attention(
#         query=query,
#         key=key,
#         value=value,
#         slot_mapping=slot_mapping,
#         block_tables=block_tables,
#         batch_valid_length=batch_valid_length,
#         context_lens_tensor=context_lens_tensor,
#         q_seq_lens=q_seq_lens,               # <- now guaranteed non-None (ones) for paged tiler
#         actual_seq_qlen=actual_seq_qlen,     # <- set when MF_FA_PREFIX_ONLY=1
#         actual_seq_kvlen=actual_seq_kvlen,   # <- set when MF_FA_PREFIX_ONLY=1
#         attn_mask=attention_mask,
#         key_cache=key_cache,
#         value_cache=value_cache)
# else:
#     core_attn_out = self.core_attention(query, key, value, attention_mask)

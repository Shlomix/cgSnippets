# ============================================================
# [1] ADD NEAR THE TOP IMPORTS (alongside existing imports)
# ============================================================
import os  # NEW


# ============================================================
# [2] INSIDE class Attention.__init__(...) AFTER your existing fields are set
#     (right after self.linear_proj is built is a fine spot)
# ============================================================
# Toggle to *intentionally* demonstrate FA's limitation:
#   When set to 1, we will pass kv_len = prefix only (i.e., < q_len), which FlashAttention rejects.
#   Enable when you want the FA failure:  export MF_FA_PREFIX_ONLY=1
self._fa_prefix_only = (os.getenv("MF_FA_PREFIX_ONLY", "0") == "1")

# Optional toggle to ignore the 2D/chunk path at this layer (treat as simple prefill/decode):
#   export MF_DISABLE_CHUNK_PARALLEL=1
self._disable_chunk_parallel = (os.getenv("MF_DISABLE_CHUNK_PARALLEL", "0") == "1")

# Graph-safe printer so you can see which path the call is taking (works in graph mode):
self._pp = ops.Print()


# ============================================================
# [3] IN Attention.construct(...), INSERT THIS PREP BLOCK JUST BEFORE
#     the existing `if self.use_flash_attention:` call to self.core_attention(...)
# ============================================================

# ---- A) Sanitize control-tensor dtypes expected by the kernels (int32) ----
if batch_valid_length is not None and batch_valid_length.dtype != mint.int32:
    batch_valid_length = mint.astype(batch_valid_length, mint.int32)

if q_seq_lens is not None and q_seq_lens.dtype != mint.int32:
    q_seq_lens = mint.astype(q_seq_lens, mint.int32)

if block_tables is not None and block_tables.dtype != mint.int32:
    block_tables = mint.astype(block_tables, mint.int32)

if slot_mapping is not None and slot_mapping.dtype != mint.int32:
    slot_mapping = mint.astype(slot_mapping, mint.int32)

# ---- B) Optionally disable the 2D/chunk path at this layer (simple flow) ----
if self._disable_chunk_parallel and q_seq_lens is not None:
    self._pp("ATTN: MF_DISABLE_CHUNK_PARALLEL=1 → ignoring q_seq_lens at this layer.")
    q_seq_lens = None

# ---- C) Guarantee the PagedAttention tiler *always* sees a q_seq_lens vector ----
# For decode, a vector of ones is valid. Without this, the paged kernel can fail during tiling.
if q_seq_lens is None and batch_valid_length is not None:
    q_seq_lens = mint.full_like(batch_valid_length, 1)  # int32, shape [B]

# ---- D) Prepare ragged lengths for FlashAttention in 2D/chunk prefill ----
# We *don’t* force FA here; we only populate the optional ragged lengths so that
# when the FA path is active for chunk prefill, it can use them correctly.
# (First prefill remains as your core implements; passing lengths on first chunk is safe —
#  q_len == kv_len in that case.)
if q_seq_lens is not None:
    # actual query length is the current chunk sizes
    actual_seq_qlen = q_seq_lens if actual_seq_qlen is None else actual_seq_qlen
    # actual KV length:
    #  - normal: prefix + chunk  (batch_valid_length + q_seq_lens)
    #  - demo:   prefix only     (batch_valid_length) -> kv_len < q_len (FA will reject, by design)
    if actual_seq_kvlen is None and batch_valid_length is not None:
        actual_seq_kvlen = batch_valid_length if self._fa_prefix_only else (batch_valid_length + q_seq_lens)
        if self._fa_prefix_only:
            self._pp("ATTN[FA_DEMO]: kv_len < q_len → expect FlashAttention to error.")
            self._pp(actual_seq_qlen)   # q_len per batch
            self._pp(actual_seq_kvlen)  # kv_len per batch (prefix only)
else:
    # No 2D lengths available; leave actual_seq_* as provided by caller.
    pass

# ---- E) Small marker so you can see the call category in logs ----
if self.use_flash_attention and q_seq_lens is not None:
    self._pp("ATTN[CALL]: FA-eligible (2D/chunk lengths present).")
else:
    self._pp("ATTN[CALL]: paged/regular path.")

# (Now let the *existing* call proceed exactly as your file already has it, e.g.)
# if self.use_flash_attention:
#     core_attn_out = self.core_attention(
#         query=query,
#         key=key,
#         value=value,
#         slot_mapping=slot_mapping,
#         block_tables=block_tables,
#         batch_valid_length=batch_valid_length,
#         context_lens_tensor=context_lens_tensor,
#         q_seq_lens=q_seq_lens,               # <- guaranteed non-None for paged tiler (ones on decode)
#         actual_seq_qlen=actual_seq_qlen,     # <- provided for FA 2D/chunk
#         actual_seq_kvlen=actual_seq_kvlen,   # <- provided for FA 2D/chunk
#         attn_mask=attention_mask,
#         key_cache=key_cache,
#         value_cache=value_cache)
# else:
#     core_attn_out = self.core_attention(query, key, value, attention_mask)

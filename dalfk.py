# =========================== COPY EVERYTHING BELOW ===========================
# === FILE: research/qwen2_5/infer/transformer.py ============================

# ---------------- PATCH 1 ----------------
# ParallelTransformerLayer.construct(...)
# INSERT THIS BLOCK **RIGHT AFTER**:
#     norm_output = self.attention_norm(x)
# and **BEFORE** the call:
#     attention_output = self.attention(norm_output, batch_valid_length, block_tables, ...)

# ==== [DBG] layer-call context (graph-safe prints) ====
print("[LAYER]", int(self.layer_index))
print("first_iter", bool(self.attention.is_first_iteration),
      "use_past", bool(self.attention.use_past),
      "use_fa",   bool(self.attention.use_flash_attention))
_x = norm_output  # this is what we pass into attention
if len(_x.shape) == 3:
    print("x_rank", 3, "B", int(_x.shape[0]), "S", int(_x.shape[1]), "H", int(_x.shape[2]))
else:
    print("x_rank", 2, "T", int(_x.shape[0]), "H", int(_x.shape[1]))
if batch_valid_length is not None:
    print("batch_valid_len_len", int(batch_valid_length.shape[0]))
if block_tables is not None:
    print("block_tables_shape", int(block_tables.shape[0]), int(block_tables.shape[1]))
# =====================================================


# ---------------- PATCH 2 ----------------
# ParallelAttention.construct(...)
# All snippets below go **INSIDE** the existing `if self.use_past:` block.

# --- PATCH 2A ---
# PLACE THIS **IMMEDIATELY AFTER Q,K,V ARE COMPUTED** (right after c_attn(x) & split)
# ==== [DBG][PA] entered use_past, QKV ranks right after projection/split ====
print("[PA] enter", True, "first_iter", bool(self.is_first_iteration))
if len(x.shape) == 3:
    print("x_rank", 3, "B", int(x.shape[0]), "S", int(x.shape[1]), "H", int(x.shape[2]))
else:
    print("x_rank", 2, "T", int(x.shape[0]), "H", int(x.shape[1]))
print("q_rank", len(query.shape),
      "q0", int(query.shape[0]) if len(query.shape)>0 else -1,
      "q1", int(query.shape[1]) if len(query.shape)>1 else -1)
print("k_rank", len(key.shape),
      "k0", int(key.shape[0]) if len(key.shape)>0 else -1,
      "k1", int(key.shape[1]) if len(key.shape)>1 else -1,
      "k2", int(key.shape[2]) if len(key.shape)>2 else -1,
      "k3", int(key.shape[3]) if len(key.shape)>3 else -1)
print("v_rank", len(value.shape),
      "v0", int(value.shape[0]) if len(value.shape)>0 else -1,
      "v1", int(value.shape[1]) if len(value.shape)>1 else -1,
      "v2", int(value.shape[2]) if len(value.shape)>2 else -1,
      "v3", int(value.shape[3]) if len(value.shape)>3 else -1)
# ============================================================================


# --- PATCH 2B ---
# PLACE THIS **JUST ABOVE** the cache write call:
#     key_out = self.paged_attention_mgr(key, value, slot_mapping, batch_valid_length, key_cache=..., value_cache=...)
# ---- [DBG][PA] about to write KV into paged cache ----
if slot_mapping is not None:
    print("slot_map_rank", len(slot_mapping.shape),
          "slot0", int(slot_mapping.shape[0]) if len(slot_mapping.shape)>0 else -1,
          "slot1", int(slot_mapping.shape[1]) if len(slot_mapping.shape)>1 else -1)
if batch_valid_length is not None:
    print("valid_len_len", int(batch_valid_length.shape[0]))
if block_tables is not None:
    print("block_tbl", int(block_tables.shape[0]), int(block_tables.shape[1]))
# ------------------------------------------------------


# --- PATCH 2C ---
# PLACE THIS **IMMEDIATELY AFTER**:
#     key_out = self.paged_attention_mgr(...)
#     query = ops.depend(query, key_out)
# ---- [DBG][PA] KV written ----
print("kv_written", True)
# --------------------------------


# --- PATCH 2D ---
# FIRST-ITERATION BRANCH: place right under `if self.is_first_iteration:`
# ---- [DBG][PA] first-iteration path ----
print("PA_first_iter", True, "use_fa", bool(self.use_flash_attention))
if self.use_flash_attention:
    print("FA1_Q", len(query.shape), int(query.shape[0]), int(query.shape[-1]))
    print("FA1_K", len(key.shape),   int(key.shape[0]),   int(key.shape[-1]))
    print("FA1_V", len(value.shape), int(value.shape[0]), int(value.shape[-1]))
    if (q_seq_lens is not None) and (batch_valid_length is not None):
        print("len_vecs", int(q_seq_lens.shape[0]), int(batch_valid_length.shape[0]))
# -----------------------------------------


# --- PATCH 2E ---
# LATER-PHASE / DECODE BRANCH:
# place at the start of the `else:` branch that would run paged attention
# (i.e., right before calling paged_attn on query)
# ---- [DBG][PA] later prefill / decode path (paged attention) ----
print("PA_later", True)
print("Q_before_paged", len(query.shape),
      int(query.shape[0]) if len(query.shape)>0 else -1,
      int(query.shape[1]) if len(query.shape)>1 else -1)
if batch_valid_length is not None:
    print("valid_len_len", int(batch_valid_length.shape[0]))
if block_tables is not None:
    print("blk_tbl", int(block_tables.shape[0]), int(block_tables.shape[1]))
# -----------------------------------------------------------------


# --- PATCH 2F ---
# RIGHT BEFORE THE OUTPUT PROJECTION (i.e., just before projecting `context_layer`)
# ---- [DBG][PA] context for projection ----
_ctx = context_layer if "context_layer" in locals() else query
if len(_ctx.shape) == 3:
    print("ctx_rank", 3, int(_ctx.shape[0]), int(_ctx.shape[1]), int(_ctx.shape[2]))
else:
    print("ctx_rank", len(_ctx.shape),
          int(_ctx.shape[0]) if len(_ctx.shape)>0 else -1,
          int(_ctx.shape[1]) if len(_ctx.shape)>1 else -1)
# -------------------------------------------
# =========================== COPY BLOCK END ===========================

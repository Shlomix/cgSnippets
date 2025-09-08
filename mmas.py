# ---------- GQA head align IN TH (tile KV heads 4 -> 28 so hidden matches Q: 512 -> 3584) ----------
nh = int(self.num_heads_per_partition)   # 28
d  = self.head_dim                       # 128
Hq = int(query.shape[-1])                # 3584 for you
kv_heads = int(k_th.shape[-1]) // d      # 4

if self.use_gqa and kv_heads != nh:
    rep = nh // max(kv_heads, 1)         # 28/4 = 7
    # [Tk, Hk] -> [Tk, n_kv, d]
    k4 = self.reshape(k_th, (B * KV, kv_heads, d))
    v4 = self.reshape(v_th, (B * KV, kv_heads, d))
    # *** IMPORTANT: use device tile, NOT mint.repeat_interleave ***
    k4 = ops.tile(k4, (1, rep, 1))       # [Tk, 28, 128]
    v4 = ops.tile(v4, (1, rep, 1))       # [Tk, 28, 128]
    # back to TH with hidden = nh*d (3584)
    k_th = self.reshape(k4, (B * KV, nh * d))
    v_th = self.reshape(v4, (B * KV, nh * d))
    Hq = nh * d

# ---------- Sanitize FA inputs: same dtype + colocate all control tensors on Q's stream ----------
# Enforce compute dtype (BF16) right at call site
if query.dtype != self.compute_dtype: query = self.cast(query, self.compute_dtype)
if k_th.dtype  != self.compute_dtype: k_th  = self.cast(k_th,  self.compute_dtype)
if v_th.dtype  != self.compute_dtype: v_th  = self.cast(v_th,  self.compute_dtype)

# Length vectors must be int32 and *device-colocated* with query
if q_seq_lens.dtype != mstype.int32:
    q_seq_lens = self.cast(q_seq_lens, mstype.int32)
if batch_valid_length.dtype != mstype.int32:
    batch_valid_length = self.cast(batch_valid_length, mstype.int32)

# Strong colocation: make them data-dependent on Q (prevents heterogeneous copy into FA)
q_seq_lens         = ops.depend(q_seq_lens, query)
batch_valid_length = ops.depend(batch_valid_length, query)

# Avoid masks on this path; let causal FA handle it internally
_attn_mask  = None
_alibi_mask = None
_prefix     = None
_padding    = None

# Also ensure K/V are evaluated on the same stream as Q before the call
k_th = ops.depend(k_th, query)
v_th = ops.depend(v_th, query)

# ---------- FlashAttention (TH) with correct wrapper arg order ----------
context_th = self.flash_attention(
    query,             # Q: [B*S_cur, Hq]
    k_th,              # K: [B*KV,   Hq]
    v_th,              # V: [B*KV,   Hq]
    _attn_mask,        # attn_mask  (wrapper arg #4)
    _alibi_mask,       # alibi_mask (wrapper arg #5)
    _prefix,           # prefix     (wrapper arg #6)
    _padding,          # padding    (wrapper arg #7)
    q_seq_lens,        # actual_seq_qlen  (int32, shape [B])
    batch_valid_length # actual_seq_kvlen (int32, shape [B])
)

# Back to [B, S, H] and early exit
context_layer = self.reshape(context_th, (B, S_cur, Hq))
attn_out = self.o_proj(context_layer)   # or self.wo if that’s your name
return attn_out

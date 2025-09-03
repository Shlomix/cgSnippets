# ====== inside research/qwen2_5/infer/transformer.py (attention class) ======
from mindspore import ops, Tensor, dtype as mstype

# --- in __init__ (replace the single FlashAttention with two) ---
if self.use_flash_attention:
    # prefill (S>1): use BNSD kernel
    self.fa_prefill = FlashAttention(
        head_num=self.num_heads_per_partition,
        scale_value=1.0 / self.norm_factor,
        next_tokens=0,
        input_layout="BNSD",
    )
    # decode (S==1): TH kernel (unchanged behavior)
    self.fa_decode = FlashAttention(
        head_num=self.num_heads_per_partition,
        scale_value=1.0 / self.norm_factor,
        next_tokens=0,
        input_layout="TH",
    )
else:
    self.fa_prefill = None
    self.fa_decode  = None

# --- tiny helpers (inside the class) ---
def _bsh_to_bnsd(self, x, n_heads):
    """[B,S,H] -> [B,N,S,D] for FlashAttention(input_layout='BNSD')."""
    B, S, H = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    D = H // int(n_heads)
    # B,S,H -> B,S,N,D -> B,N,S,D
    return ops.transpose(ops.reshape(x, (B, S, n_heads, D)), (0, 2, 1, 3))

def _gather_kv_from_cache_inline(self, block_tables):
    """Contiguous K/V from manager cache; static shapes only."""
    kc = self.paged_attention_mgr.key_cache
    vc = self.paged_attention_mgr.value_cache
    B  = int(block_tables.shape[0])
    M  = int(block_tables.shape[1])
    bs = int(kc.shape[1])
    Hk = int(kc.shape[-1])
    flat = ops.reshape(block_tables, (B * M,))
    k_full = ops.reshape(ops.gather(kc, flat, 0), (B, M * bs, Hk))
    v_full = ops.reshape(ops.gather(vc, flat, 0), (B, M * bs, Hk))
    return k_full, v_full, (M * bs)

# --- in construct(...), keep the manager write + depend exactly as-is ---
# key_out = self.paged_attention_mgr(key, value, slot_mapping, batch_valid_length, ...)
# query   = ops.depend(query, key_out)

# --- right after the depend barrier, handle later-chunk prefill with FA ---
S_cur = int(getattr(query, "shape", (0,1,0))[1] or 1)
is_chunked = (S_cur > 1) and (q_seq_lens is not None)

if (not self.is_first_iteration) and self.use_flash_attention and is_chunked:
    # 1) read K/V from cache (now contains prefix+this chunk)
    k_full, v_full, KV_CAP = self._gather_kv_from_cache_inline(block_tables)

    # 2) quick safety: hidden sizes must match Q; otherwise, fall back to paged
    Hq = int(query.shape[-1])
    Hk = int(k_full.shape[-1])
    if Hq != Hk:
        # GQA case and you want minimal changes -> just keep paged attention
        # (Optionally, you can tile KV heads later to force equality.)
        context_layer = self.paged_attention_mgr.paged_attn(
            query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=q_seq_lens
        )
        return context_layer

    # 3) convert Q/K/V from BSH -> BNSD and call prefill FA
    q_bnsd = self._bsh_to_bnsd(query, self.num_heads_per_partition)
    k_bnsd = self._bsh_to_bnsd(k_full, self.num_heads_per_partition)
    v_bnsd = self._bsh_to_bnsd(v_full, self.num_heads_per_partition)

    # Let FA derive the staircase using (q_seq_lens, batch_valid_length) — same as first-iter path.
    context_layer = self.fa_prefill(
        q_bnsd, k_bnsd, v_bnsd, attn_mask, alibi_mask, None, None, q_seq_lens, batch_valid_length
    )
    return context_layer

# ...everything else stays exactly as before...
# - first iteration: use your existing FA call (you can switch it to self.fa_prefill too if you want)
# - decode/non-chunked: original path (paged attention), unchanged

# --- right after the manager write + ops.depend(...), inside:
# if (not self.is_first_iteration) and self.use_flash_attention and is_chunked:

kc = getattr(self.paged_attention_mgr, "key_cache", None)
vc = getattr(self.paged_attention_mgr, "value_cache", None)

if (kc is None) or (vc is None):
    # Cache not ready yet -> do chunk-only FA and return (avoids Gather on None)
    def _bsh_to_bnsd(x, n_heads):
        B, S, H = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        D = H // int(n_heads)
        return ops.transpose(ops.reshape(x, (B, S, n_heads, D)), (0, 2, 1, 3))  # B,N,S,D

    q_bnsd = _bsh_to_bnsd(query, self.num_heads_per_partition)
    k_bnsd = _bsh_to_bnsd(key,   self.num_heads_per_partition)
    v_bnsd = _bsh_to_bnsd(value, self.num_heads_per_partition)

    context_layer = self.fa_prefill(
        q_bnsd, k_bnsd, v_bnsd,
        attn_mask, alibi_mask, None, None, q_seq_lens, batch_valid_length
    )
    return context_layer

# ---- cache is ready: do the gather safely (static shapes only)
B  = int(block_tables.shape[0])
M  = int(block_tables.shape[1])
bs = int(kc.shape[1])          # cache layout: [num_blocks, block_size, Hk]
Hk = int(kc.shape[-1])

flat   = ops.reshape(block_tables, (B * M,))                 # (B*M,)
k_full = ops.reshape(ops.gather(kc, flat, 0), (B, M*bs, Hk)) # (B, KV_CAP, Hk)
v_full = ops.reshape(ops.gather(vc, flat, 0), (B, M*bs, Hk))

# (optionally: check hidden dims; if Hk != query.shape[-1], fall back to paged)
Hq = int(query.shape[-1])
if Hk != Hq:
    context_layer = self.paged_attention_mgr.paged_attn(
        query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=q_seq_lens
    )
    return context_layer

# Convert BSH -> BNSD for the prefill FA and call it
q_bnsd = ops.transpose(ops.reshape(query, (B, int(query.shape[1]), self.num_heads_per_partition, Hq // self.num_heads_per_partition)), (0, 2, 1, 3))
k_bnsd = ops.transpose(ops.reshape(k_full, (B, M*bs, self.num_heads_per_partition, Hq // self.num_heads_per_partition)), (0, 2, 1, 3))
v_bnsd = ops.transpose(ops.reshape(v_full, (B, M*bs, self.num_heads_per_partition, Hq // self.num_heads_per_partition)), (0, 2, 1, 3))

context_layer = self.fa_prefill(
    q_bnsd, k_bnsd, v_bnsd,
    attn_mask, alibi_mask, None, None, q_seq_lens, batch_valid_length
)
return context_layer

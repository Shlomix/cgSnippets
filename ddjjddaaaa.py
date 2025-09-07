            # always write K/V to cache (unchanged)
            key_out = self.paged_attention_mgr(key, value, slot_mapping, batch_valid_length,
                                               key_cache=key_cache, value_cache=value_cache)
            query = ops.depend(query, key_out)

            # use S to decide: prefill (S>1) vs decode (S==1)
            bs, seq_len, _ = x.shape
            if self.use_flash_attention and seq_len > 1:
                # -------- Prefill (first + later) with FlashAttention ----------

                if self.is_first_iteration:
                    # your original FA first-chunk path (kept verbatim)
                    context_layer = self.flash_attention(query, key, value, attn_mask, alibi_mask, None, None,
                                                         q_seq_lens, batch_valid_length)
                else:
                    # later prefill chunks: read all KV from cache, pack to BNSD, run FA
                    # block_tables: [B, M], key_cache/value_cache indexed as [B*M, block, Hk]
                    if block_tables.dtype != mstype.int32:
                        block_tables = ops.Cast()(block_tables, mstype.int32)
                    B, M = int(block_tables.shape[0]), int(block_tables.shape[1])
                    flat = ops.Reshape()(block_tables, (B * M,))  # [B*M]

                    k_raw = ops.Gather()(key_cache,   flat, 0)    # [B*M, bs_block, Hk]
                    v_raw = ops.Gather()(value_cache, flat, 0)
                    bs_block = int(k_raw.shape[1]); Hk = int(k_raw.shape[-1])
                    KV = M * bs_block

                    # [B*M, bs, Hk] -> [B, KV, Hk]
                    k_bsh = ops.Reshape()(k_raw, (B, KV, Hk))
                    v_bsh = ops.Reshape()(v_raw, (B, KV, Hk))

                    # GQA align: tile KV heads to match Q heads if needed (your style & symbols)
                    nh = int(self.num_heads_per_partition); d = self.head_dim
                    kv_heads = Hk // d
                    if self.use_gqa and kv_heads != nh:
                        k4 = ops.Reshape()(k_bsh, (B, KV, kv_heads, d))
                        v4 = ops.Reshape()(v_bsh, (B, KV, kv_heads, d))
                        k4 = mint.repeat_interleave(k4, repeats=self.repeat_num, dim=2)
                        v4 = mint.repeat_interleave(v4, repeats=self.repeat_num, dim=2)
                        k_bsh = ops.Reshape()(k4, (B, KV, nh * d))
                        v_bsh = ops.Reshape()(v4, (B, KV, nh * d))

                    # Pack to BNSD using your own shape comments:
                    # Q: [B, S, H] -> [B, S, N, D] -> [B, N, S, D]
                    q_nd   = query.reshape(bs, seq_len, nh, d)
                    q_bnsd = q_nd.transpose(0, 2, 1, 3)
                    # K/V: [B, KV, H] -> [B, KV, N, D] -> [B, N, KV, D]
                    k_nd   = k_bsh.reshape(B, KV, nh, d)
                    v_nd   = v_bsh.reshape(B, KV, nh, d)
                    k_bnsd = k_nd.transpose(0, 2, 1, 3)
                    v_bnsd = v_nd.transpose(0, 2, 1, 3)

                    # minimal dtype guard: match K/V to Q dtype right at FA
                    fa_dtype = q_bnsd.dtype
                    if k_bnsd.dtype != fa_dtype: k_bnsd = ops.Cast()(k_bnsd, fa_dtype)
                    if v_bnsd.dtype != fa_dtype: v_bnsd = ops.Cast()(v_bnsd, fa_dtype)
                    if (alibi_mask is not None) and (alibi_mask.dtype != fa_dtype):
                        alibi_mask = ops.Cast()(alibi_mask, fa_dtype)

                    context_layer = self.flash_attention_prefill_bnsd(
                        q_bnsd, k_bnsd, v_bnsd,
                        None,                  # attn_mask not needed with causal kernels
                        alibi_mask,
                        None, None,
                        q_seq_lens,            # actual_seq_qlen
                        batch_valid_length     # actual_seq_kvlen
                    )
                    # [B, N, S, D] -> [B, S, H]
                    context_layer = context_layer.transpose(0, 2, 1, 3).reshape(
                        bs, seq_len, self.hidden_size_per_partition)

            else:
                # -------- Decode (S==1) or FA disabled: stock paged attention ----------
                context_layer = self.paged_attention_mgr.paged_attn(
                    query, batch_valid_length, block_tables,
                    attn_mask, q_seq_lens, key_cache, value_cache
                )

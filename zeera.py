def construct(self,
              query,
              key,
              value,
              slot_mapping=None,
              block_tables=None,
              batch_valid_length=None,
              context_lens_tensor=None,
              q_seq_lens=None,
              actual_seq_qlen=None,
              actual_seq_kvlen=None,
              attn_mask=None,
              padding_mask=None,
              prefix=None,
              key_cache=None,
              value_cache=None):

    # 0) Always write current chunk into paged cache so PA works unconditionally.
    if not self.use_multi_latent_attention:
        self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    # 0.1) If any mask is present, warn once and drop it for FA (TND).
    if attn_mask is not None:
        try:
            print("[FA] dropping attn_mask for FlashAttention; got shape:", attn_mask.shape)
        except Exception:
            print("[FA] dropping attn_mask for FlashAttention")
    # Keep original attn_mask for PA later, but NEVER pass to FA.
    _fa_padding_mask = None
    _fa_attn_mask = None

    # Helper: reshape TH->[T,N,D]
    n_q = int(self.head_num)
    n_kv = int(self.kv_head_num) if self.kv_head_num is not None else int(self.head_num)
    d = int(self.hidden_size_per_attention_head)

    def _to_tnd(x_2d, n_heads, hd):
        return self.reshape(x_2d, (-1, n_heads, hd))

    def _harmonize(q, k, v, cache):
        if cache is not None:
            dt = cache.dtype
            q = self.cast(q, dt)
            k = self.cast(k, dt)
            v = self.cast(v, dt)
        return q, k, v

    # 1) First prefill: FlashAttention (TND) on current chunk (no masks)
    if self.is_prefill:
        q_tnd = _to_tnd(query, n_q, d)
        k_tnd = _to_tnd(key,   n_kv, d)
        v_tnd = _to_tnd(value, n_kv, d)
        q_tnd, k_tnd, v_tnd = _harmonize(q_tnd, k_tnd, v_tnd, key_cache)

        # FA call: (q, k, v, alibi, rel_shift, padding_mask, attn_mask, prefix, actual_q, actual_kv)
        _, _, _, ctx = self.flash_attention(
            q_tnd, k_tnd, v_tnd,
            None, None,
            _fa_padding_mask, _fa_attn_mask,
            prefix, actual_seq_qlen, actual_seq_kvlen
        )
        # Optional debug: print("FA 1st prefill TND:", q_tnd.shape, k_tnd.shape, v_tnd.shape, "->", ctx.shape)
        return ctx

    # 2) Second+ prefill: gather contiguous KV from cache, then FA (TND) with NO masks
    is_second_plus = False
    if q_seq_lens is not None:
        try:
            from mindspore import ops as _ops
            mx = self.reduce_max(q_seq_lens)
            is_second_plus = bool(_ops.Greater()(mx, self.scalar_to_tensor(1, q_seq_lens.dtype)).asnumpy().item())
        except Exception:
            is_second_plus = True  # best effort

    if is_second_plus and (key_cache is not None) and (value_cache is not None) \
       and (block_tables is not None) and (actual_seq_kvlen is not None):
        # Total KV length from last element
        total_kv_len = int(actual_seq_kvlen.asnumpy()[-1])

        # Gather contiguous KV from cache -> [Tk, Nkv, D]
        num_blocks, block_size, kv_heads, hd = map(int, key_cache.shape)
        need = (total_kv_len + block_size - 1) // block_size
        blocks_1b = block_tables[0][:need]
        blocks_0b = blocks_1b - ops.ones_like(blocks_1b)
        gathered_k = self.gather(key_cache, blocks_0b, 0)   # [need, block, Nkv, D]
        gathered_v = self.gather(value_cache, blocks_0b, 0) # [need, block, Nkv, D]
        k_full = self.reshape(gathered_k, (-1, kv_heads, hd))[:total_kv_len]
        v_full = self.reshape(gathered_v, (-1, kv_heads, hd))[:total_kv_len]

        q_tnd = _to_tnd(query, n_q, d)
        q_tnd, k_full, v_full = _harmonize(q_tnd, k_full, v_full, key_cache)

        _, _, _, ctx = self.flash_attention(
            q_tnd, k_full, v_full,
            None, None,
            _fa_padding_mask, _fa_attn_mask,
            prefix, actual_seq_qlen, actual_seq_kvlen
        )
        # Optional debug: print("FA 2D+ prefill TND:", q_tnd.shape, k_full.shape, v_full.shape, "->", ctx.shape)
        return ctx

    # 3) Decode / fallback: PagedAttention (keeps original mask)
    if self.use_multi_latent_attention:
        ctx = self.paged_attention(
            query, key_cache, key_cache,
            block_tables, batch_valid_length,
            None, None, attn_mask, q_seq_lens
        )
    else:
        if key_cache is not None:
            query = self.cast(query, key_cache.dtype)
        ctx = self.paged_attention(
            query, key_cache, value_cache,
            block_tables, batch_valid_length,
            None, None, attn_mask, q_seq_lens
        )
    return ctx

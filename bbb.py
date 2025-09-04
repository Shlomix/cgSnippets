# --- FA prefill fast-path (drop-in replacement for your current block) ---
# preconditions: we've already executed the manager write + ops.depend barrier
if getattr(self, "use_flash_attention", False) and (self.fa_prefill is not None) and (q_seq_lens is not None):
    kc = self.paged_attention_mgr.key_cache
    vc = self.paged_attention_mgr.value_cache
    if (kc is not None) and (vc is not None):
        # gather K/V from cache as contiguous [B, KV_CAP, Hk]
        B  = int(block_tables.shape[0])
        M  = int(block_tables.shape[1])
        bs = int(kc.shape[1])               # [num_blocks, block_size, Hk]
        Hk = int(kc.shape[-1])

        flat   = ops.reshape(block_tables, (B * M,))
        k_full = ops.reshape(ops.gather(kc, flat, 0), (B, M * bs, Hk))
        v_full = ops.reshape(ops.gather(vc, flat, 0), (B, M * bs, Hk))
        KV     = M * bs

        # ---- read query shape robustly and normalise to BSH ----
        def _as_bsh(x):
            shp = getattr(x, "shape", ())
            r = len(shp)
            if r == 2:  # TH
                T, H = int(shp[0]), int(shp[1])
                return 1, T, H, ops.reshape(x, (1, T, H))      # B=1, S=T
            if r == 3:  # BSH
                Bx, Sx, Hx = int(shp[0]), int(shp[1]), int(shp[2])
                return Bx, Sx, Hx, x
            if r == 4:  # BNSD
                Bx, Nx, Sx, Dx = int(shp[0]), int(shp[1]), int(shp[2]), int(shp[3])
                return Bx, Sx, Nx * Dx, ops.reshape(ops.transpose(x, (0, 2, 1, 3)), (Bx, Sx, Nx * Dx))
            raise RuntimeError("unsupported rank for query")

        Bq, S_cur, Hq, q_bsh = _as_bsh(query)

        # ---- form heads and expand KV to match Q heads (handles GQA) ----
        nh = int(self.num_heads_per_partition)
        if (nh > 0) and (Hq % nh == 0):
            d = Hq // nh
            if (Hk % d) == 0:
                kv_heads = Hk // d
                if kv_heads != nh:
                    rep = nh // max(kv_heads, 1)     # tile KV heads to Q heads
                    k4 = ops.reshape(k_full, (B, KV, kv_heads, d))
                    v4 = ops.reshape(v_full, (B, KV, kv_heads, d))
                    k4 = ops.tile(k4, (1, 1, rep, 1))
                    v4 = ops.tile(v4, (1, 1, rep, 1))
                    k_full = ops.reshape(k4, (B, KV, nh * d))
                    v_full = ops.reshape(v4, (B, KV, nh * d))
                # now hidden dims match Q: Hk == Hq == nh * d

                # ---- convert to BNSD and call FA (prefill) ----
                q_bnsd = ops.transpose(ops.reshape(q_bsh,  (Bq, S_cur, nh, d)), (0, 2, 1, 3))
                k_bnsd = ops.transpose(ops.reshape(k_full, (B,  KV,   nh, d)), (0, 2, 1, 3))
                v_bnsd = ops.transpose(ops.reshape(v_full, (B,  KV,   nh, d)), (0, 2, 1, 3))

                context_layer = self.fa_prefill(
                    q_bnsd, k_bnsd, v_bnsd,
                    attn_mask, alibi_mask, None, None,
                    q_seq_lens, batch_valid_length
                )
                return context_layer
# if we didn’t return, fall through to your original branch (paged/core)

# ===== FA-prefill block (graph-mode safe) =====
from mindspore import ops, Tensor, dtype as mstype

# Ensure integer dtypes are consistent for routing tensors
if block_tables.dtype != mstype.int32:
    block_tables = ops.cast(block_tables, mstype.int32)
if batch_valid_length.dtype != mstype.int32:
    batch_valid_length = ops.cast(batch_valid_length, mstype.int32)

use_fa_prefill = bool(getattr(self, "use_flash_attention", False) and (q_seq_lens is not None))

if use_fa_prefill:
    kc, vc = key_cache, value_cache  # external caches
    if (kc is not None) and (vc is not None):
        # ---------- Safe gather of valid KV blocks ----------
        B  = int(block_tables.shape[0])
        M  = int(block_tables.shape[1])
        bs = int(kc.shape[1])                 # block_size
        Hk = int(kc.shape[-1])

        # valid blocks per batch: ceil(L/bs)
        L  = batch_valid_length                                   # [B] int32
        m_per_batch = ops.floor_div(ops.add(L, Tensor(bs - 1, mstype.int32)),
                                    Tensor(bs, mstype.int32))     # [B] int32

        idx = ops.arange(0, M, 1)                                 # [M] (int64)
        idx = ops.cast(idx, mstype.int32)
        valid_mask = ops.lt(ops.expand_dims(idx, 0), ops.expand_dims(m_per_batch, 1))  # [B,M] bool
        safe_bt = ops.where(valid_mask, block_tables, Tensor(0, mstype.int32))         # [B,M]

        flat = ops.reshape(safe_bt, (B * M,))                     # [B*M]
        k_raw = ops.gather(kc, flat, 0)                           # [B*M, bs, Hk]
        v_raw = ops.gather(vc, flat, 0)                           # [B*M, bs, Hk]
        k_full = ops.reshape(k_raw, (B, M * bs, Hk))              # [B, M*bs, Hk]
        v_full = ops.reshape(v_raw, (B, M * bs, Hk))

        max_m = int(ops.reduce_max(m_per_batch).asnumpy())
        KV    = max_m * bs
        k_full = k_full[:, :KV, :]
        v_full = v_full[:, :KV, :]

        # ---------- Normalize Q to BSH & get S_cur/Hq (single scope) ----------
        shp = tuple(getattr(query, "shape", ()))
        Bq = 1
        if len(shp) == 2:                 # TH: (T,Hq) -> (1,T,Hq)
            S_cur = int(shp[0])
            Hq    = int(shp[1])
            q_bsh = ops.reshape(query, (1, S_cur, Hq))
        elif len(shp) == 3:               # BSH: (B,S,Hq)
            Bq    = int(shp[0])
            S_cur = int(shp[1])
            Hq    = int(shp[2])
            q_bsh = query
        else:
            q_bsh = None  # unexpected layout (e.g., already BNSD)

        if q_bsh is not None:
            nh = int(self.num_heads_per_partition)
            if (nh > 0) and (Hq % nh == 0):
                d = Hq // nh

                # ---------- Handle GQA: tile KV heads up to nh ----------
                if (Hk % d) == 0:
                    kv_heads = Hk // d
                    if kv_heads != nh:
                        rep = nh // max(kv_heads, 1)
                        k4 = ops.reshape(k_full, (B, KV, kv_heads, d))
                        v4 = ops.reshape(v_full, (B, KV, kv_heads, d))
                        k4 = ops.tile(k4, (1, 1, rep, 1))
                        v4 = ops.tile(v4, (1, 1, rep, 1))
                        k_full = ops.reshape(k4, (B, KV, nh * d))
                        v_full = ops.reshape(v4, (B, KV, nh * d))
                        Hk = nh * d

                    # proceed only when hidden dims match
                    if Hk == Hq:
                        # ---------- Convert to BNSD & call FA (prefill) ----------
                        q_bnsd = ops.transpose(ops.reshape(q_bsh,  (Bq, S_cur, nh, d)), (0, 2, 1, 3))
                        k_bnsd = ops.transpose(ops.reshape(k_full, (B,  KV,   nh, d)), (0, 2, 1, 3))
                        v_bnsd = ops.transpose(ops.reshape(v_full, (B,  KV,   nh, d)), (0, 2, 1, 3))

                        context_layer = self.fa_prefill(
                            q_bnsd, k_bnsd, v_bnsd,
                            attn_mask, alibi_mask, None, None,
                            q_seq_lens, batch_valid_length
                        )
                        return context_layer
# ===== end FA-prefill block (falls through to original branches if not returned) =====

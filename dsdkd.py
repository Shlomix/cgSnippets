# ======================================================================
# File: research/qwen2_5/infer/parallel_paged_attention_mgr.py
# Goal: KV write path for paged cache (ReshapeAndCache) – FP16 only, no shadows
# ======================================================================

from mindspore import ops
from mindspore import dtype as mstype

class ParallelPagedAttentionMgr(nn.Cell):
    # ... keep your existing __init__ (must set self.key_cache/self.value_cache when npu_mem_size>0)

    def construct(self, key, value, slot_mapping, _, key_cache=None, value_cache=None):
        """
        Write K/V into paged KV cache. In MindFormers v1.6 the auto-generated
        ReshapeAndCache kernel expects FP16 payloads and FP16 cache buffers.
        This version avoids any shadow buffers: everything must be FP16.
        """
        use_external = (self.npu_mem_size == -1)
        tgt_kc = key_cache if use_external else self.key_cache
        tgt_vc = value_cache if use_external else self.value_cache

        # Sanity: target caches exist and are FP16
        if (tgt_kc is None) or (tgt_vc is None):
            raise RuntimeError("KV caches are None; external mode requires caller-provided buffers.")
        if (tgt_kc.dtype != mstype.float16) or (tgt_vc.dtype != mstype.float16):
            raise TypeError(
                f"ReshapeAndCache requires FP16 caches, got {tgt_kc.dtype} / {tgt_vc.dtype}. "
                "Set compute_dtype=float16 and allocate FP16 KV caches."
            )

        # Normalize inputs for the kernel
        if key.dtype != mstype.float16:
            key = ops.cast(key, mstype.float16)
        if value.dtype != mstype.float16:
            value = ops.cast(value, mstype.float16)
        # routing/indices must be int32
        if (slot_mapping is not None) and (slot_mapping.dtype != mstype.int32):
            slot_mapping = ops.cast(slot_mapping, mstype.int32)

        # v1.6 signature: (key, value, key_cache, value_cache, slot_mapping)
        return self.reshape_and_cache(key, value, tgt_kc, tgt_vc, slot_mapping)


# ======================================================================
# File: research/qwen2_5/infer/transformer.py
# Goal: Prefill uses FlashAttention reading K/V from paged cache
#       (all FP16, handles TH/BSH layouts, GQA tiling, BNSD conversion)
# ======================================================================

from mindspore import ops, Tensor
from mindspore import dtype as mstype

class ParallelTransformerLayer(nn.Cell):
    # ... your existing __init__ that builds q/k/v linears, FA, manager etc.

    # ---- helper: gather KV from paged cache and prepare for FA (BNSD) ----
    def _fa_pack_from_cache(self,
                            query_bsh,              # [B,S_cur,Hq] or [1,S_cur,Hq]
                            key_cache, value_cache, # [B*?blocks, block_size, Hk]
                            block_tables,           # [B,M] int32
                            batch_valid_length):    # [B]   int32
        # Ensure integer routing dtypes
        if block_tables.dtype != mstype.int32:
            block_tables = ops.cast(block_tables, mstype.int32)
        if batch_valid_length.dtype != mstype.int32:
            batch_valid_length = ops.cast(batch_valid_length, mstype.int32)

        # Shapes / constants
        B  = int(block_tables.shape[0])
        M  = int(block_tables.shape[1])
        bs = int(key_cache.shape[1])    # block_size

        # valid blocks per batch: ceil(L/bs)
        L  = batch_valid_length                                         # [B]
        m_per_batch = ops.floor_div(ops.add(L, Tensor(bs - 1, mstype.int32)),
                                    Tensor(bs, mstype.int32))           # [B]

        # mask invalid block ids, gather only valid blocks
        idx = ops.cast(ops.arange(0, M, 1), mstype.int32)               # [M]
        valid = ops.lt(ops.expand_dims(idx, 0), ops.expand_dims(m_per_batch, 1))  # [B,M] bool
        safe_bt = ops.where(valid, block_tables, Tensor(0, mstype.int32))         # [B,M]
        flat = ops.reshape(safe_bt, (B * M,))                                        # [B*M]

        k_raw = ops.gather(key_cache,   flat, 0)                           # [B*M, bs, Hk]
        v_raw = ops.gather(value_cache, flat, 0)                           # [B*M, bs, Hk]
        Hk = int(k_raw.shape[-1])

        k_full = ops.reshape(k_raw, (B, M * bs, Hk))                       # [B, M*bs, Hk]
        v_full = ops.reshape(v_raw, (B, M * bs, Hk))

        # truncate to actual max KV length among the batch
        max_m = int(ops.reduce_max(m_per_batch).asnumpy())
        KV = max_m * bs
        k_full = k_full[:, :KV, :]
        v_full = v_full[:, :KV, :]

        # normalize Q to BSH and extract head dims
        shp = tuple(query_bsh.shape)
        if len(shp) == 2:                     # TH -> treat as B=1
            S_cur = int(shp[0]); Hq = int(shp[1])
            Bq = 1
            q_bsh = ops.reshape(query_bsh, (1, S_cur, Hq))
        else:                                 # BSH
            Bq, S_cur, Hq = int(shp[0]), int(shp[1]), int(shp[2])
            q_bsh = query_bsh

        nh = int(self.num_heads_per_partition)
        d  = Hq // nh                          # assume divisible

        # GQA: tile KV heads up to nh if needed
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

        # Convert to BNSD for FA
        q_bnsd = ops.transpose(ops.reshape(q_bsh,  (Bq, S_cur, nh, d)), (0, 2, 1, 3))
        k_bnsd = ops.transpose(ops.reshape(k_full, (B,  KV,   nh, d)), (0, 2, 1, 3))
        v_bnsd = ops.transpose(ops.reshape(v_full, (B,  KV,   nh, d)), (0, 2, 1, 3))

        return q_bnsd, k_bnsd, v_bnsd

    # ---- inside your construct(...) where attention is computed ----
    def construct(self, x, mask, batch_valid_length, block_tables, slot_mapping,
                  freqs_cis, attn_mask=None, alibi_mask=None, q_seq_lens=None,
                  key_cache=None, value_cache=None):
        # ... your existing pre-attention norms and Q/K/V projections ...
        # Suppose you already computed query, key, value here (BSH or TH), all FP16.

        # 1) Always write this chunk’s K/V into the paged cache (FP16, int32 indices)
        if slot_mapping.dtype != mstype.int32:
            slot_mapping = ops.cast(slot_mapping, mstype.int32)
        cache_write = self.paged_attention_mgr(
            key, value, slot_mapping, batch_valid_length,
            key_cache=key_cache, value_cache=value_cache
        )
        # enforce write-before-read ordering
        query = ops.depend(query, cache_write)

        # 2) Prefill compute uses FlashAttention for ALL prefill chunks
        #    (assumes self.use_past is True for prefill; adjust predicate as per your pipeline)
        #    Gather K/V back from cache and run FA.
        q_bnsd, k_bnsd, v_bnsd = self._fa_pack_from_cache(
            query, key_cache, value_cache, block_tables, batch_valid_length
        )

        # Ensure FA sees FP16 inputs (end-to-end FP16 run)
        if q_bnsd.dtype != mstype.float16:
            q_bnsd = ops.cast(q_bnsd, mstype.float16)
        if k_bnsd.dtype != mstype.float16:
            k_bnsd = ops.cast(k_bnsd, mstype.float16)
        if v_bnsd.dtype != mstype.float16:
            v_bnsd = ops.cast(v_bnsd, mstype.float16)

        # Optional: cast masks to FP16 if FA expects that (usually not required for boolean masks)
        # if (attn_mask is not None) and (attn_mask.dtype != mstype.float16):
        #     attn_mask = ops.cast(attn_mask, mstype.float16)
        # if (alibi_mask is not None) and (alibi_mask.dtype != mstype.float16):
        #     alibi_mask = ops.cast(alibi_mask, mstype.float16)

        context_layer = self.fa_prefill(
            q_bnsd, k_bnsd, v_bnsd,
            attn_mask, alibi_mask, None, None,
            q_seq_lens, batch_valid_length
        )

        # 3) Post-attention projection & residual (your original code)
        # ...
        return context_layer  # or continue with the rest of your block as in original

# ======================================================================
# File: research/qwen2_5/infer/parallel_paged_attention_mgr.py
# Purpose: FP16-only KV write for paged cache (no shadows)
# ======================================================================

from mindspore import nn, ops
from mindspore import dtype as mstype

class ParallelPagedAttentionMgr(nn.Cell):
    # __init__ remains as in your repo; it should define:
    #   self.npu_mem_size, self.key_cache/self.value_cache when npu_mem_size>0
    #   self.reshape_and_cache (auto-generated op)
    #   self.paged_attention (auto-generated op) etc.

    def construct(self, key, value, slot_mapping, _, key_cache=None, value_cache=None):
        """
        Write K/V into paged KV cache. In MindFormers 1.6 the auto-generated
        ReshapeAndCache kernel expects FP16 payloads & FP16 cache buffers.
        """
        use_external = (self.npu_mem_size == -1)
        tgt_kc = key_cache if use_external else self.key_cache
        tgt_vc = value_cache if use_external else self.value_cache

        # caches must exist and be FP16
        if (tgt_kc is None) or (tgt_vc is None):
            raise RuntimeError("KV caches are None in paged-attention manager.")
        if (tgt_kc.dtype != mstype.float16) or (tgt_vc.dtype != mstype.float16):
            raise TypeError(f"ReshapeAndCache requires FP16 caches; got {tgt_kc.dtype} / {tgt_vc.dtype}.")

        # normalize inputs for the kernel
        if key.dtype != mstype.float16:
            key = ops.cast(key, mstype.float16)
        if value.dtype != mstype.float16:
            value = ops.cast(value, mstype.float16)
        if (slot_mapping is not None) and (slot_mapping.dtype != mstype.int32):
            slot_mapping = ops.cast(slot_mapping, mstype.int32)

        # v1.6 signature: (key, value, key_cache, value_cache, slot_mapping)
        return self.reshape_and_cache(key, value, tgt_kc, tgt_vc, slot_mapping)

# ======================================================================
# File: research/qwen2_5/infer/transformer.py
# Purpose: Use FA for chunk-prefill per policy, PA for decode; FP16 end-to-end
# ======================================================================

from mindspore import nn, ops, Tensor
from mindspore import dtype as mstype

class ParallelTransformerLayer(nn.Cell):
    def __init__(self, layer_index, config):
        super().__init__()
        self.layer_index = layer_index
        self.config = config

        # flags commonly present in the repo
        self.use_past = True
        self.use_flash_attention = True

        # policy: "later_only" (default) or "all"
        self.fa_prefill_policy = getattr(config, "fa_prefill_policy", "later_only")

        # track first-prefill pass (python bool; OK in graph because used only for gating)
        self.is_first_iteration = True

        # heads info
        self.num_heads_per_partition = config.num_heads // config.tensor_parallel
        self.head_dim = config.hidden_size // config.num_heads

        # you already have these in your code:
        # self.q_proj, self.k_proj, self.v_proj
        # self.paged_attention_mgr
        # self.fa_prefill   (FlashAttention with input_layout="BNSD")

    # ---- helper: gather KV from paged cache and convert to BNSD for FA ----
    def _fa_pack_from_cache(self, query_bsh, key_cache, value_cache, block_tables, batch_valid_length):
        if block_tables.dtype != mstype.int32:
            block_tables = ops.cast(block_tables, mstype.int32)
        if batch_valid_length.dtype != mstype.int32:
            batch_valid_length = ops.cast(batch_valid_length, mstype.int32)

        B  = int(block_tables.shape[0])
        M  = int(block_tables.shape[1])
        bs = int(key_cache.shape[1])
        Hk = int(key_cache.shape[-1])

        # valid blocks per batch: ceil(L/bs)
        L  = batch_valid_length
        m_per_batch = ops.floor_div(ops.add(L, Tensor(bs - 1, mstype.int32)),
                                    Tensor(bs, mstype.int32))   # [B]

        # gather only valid blocks
        idx = ops.cast(ops.arange(0, M, 1), mstype.int32)                  # [M]
        valid = ops.lt(ops.expand_dims(idx, 0), ops.expand_dims(m_per_batch, 1))  # [B,M]
        safe_bt = ops.where(valid, block_tables, Tensor(0, mstype.int32))         # [B,M]
        flat = ops.reshape(safe_bt, (B * M,))                                       # [B*M]

        k_raw = ops.gather(key_cache,   flat, 0)                         # [B*M, bs, Hk]
        v_raw = ops.gather(value_cache, flat, 0)                         # [B*M, bs, Hk]
        k_full = ops.reshape(k_raw, (B, M * bs, Hk))                     # [B, M*bs, Hk]
        v_full = ops.reshape(v_raw, (B, M * bs, Hk))

        # truncate to actual max KV among batch
        max_m = int(ops.reduce_max(m_per_batch).asnumpy())
        KV = max_m * bs
        k_full = k_full[:, :KV, :]
        v_full = v_full[:, :KV, :]

        # normalize Q to BSH (supports TH input too)
        shp = tuple(query_bsh.shape)
        if len(shp) == 2:
            S_cur = int(shp[0]); Hq = int(shp[1]); Bq = 1
            q_bsh = ops.reshape(query_bsh, (1, S_cur, Hq))
        else:
            Bq, S_cur, Hq = int(shp[0]), int(shp[1]), int(shp[2])
            q_bsh = query_bsh

        nh = int(self.num_heads_per_partition)
        d  = Hq // nh

        # GQA tiling if needed
        kv_heads = int(k_full.shape[-1]) // d
        if kv_heads != nh:
            rep = nh // max(kv_heads, 1)
            k4 = ops.reshape(k_full, (B, KV, kv_heads, d))
            v4 = ops.reshape(v_full, (B, KV, kv_heads, d))
            k4 = ops.tile(k4, (1, 1, rep, 1))
            v4 = ops.tile(v4, (1, 1, rep, 1))
            k_full = ops.reshape(k4, (B, KV, nh * d))
            v_full = ops.reshape(v4, (B, KV, nh * d))

        # BNSD for FA
        q_bnsd = ops.transpose(ops.reshape(q_bsh,  (Bq, S_cur, nh, d)), (0, 2, 1, 3))
        k_bnsd = ops.transpose(ops.reshape(k_full, (B,  KV,   nh, d)), (0, 2, 1, 3))
        v_bnsd = ops.transpose(ops.reshape(v_full, (B,  KV,   nh, d)), (0, 2, 1, 3))
        return q_bnsd, k_bnsd, v_bnsd, S_cur

    # ---- main forward ----
    def construct(self, x, mask, batch_valid_length, block_tables, slot_mapping,
                  freqs_cis, attn_mask=None, alibi_mask=None, q_seq_lens=None,
                  key_cache=None, value_cache=None):
        # ... norm + projections (existing code) ...
        # query = self.q_proj(...)
        # key   = self.k_proj(...)
        # value = self.v_proj(...)

        # Determine S_cur and prefill
        shp = tuple(getattr(x, "shape", ()))
        if len(shp) == 2:
            S_cur = int(shp[0])
        else:
            S_cur = int(shp[1])
        is_prefill = (S_cur > 1)

        # ---- 1) Always write this chunk’s K/V into paged cache (FP16, int32 indices) ----
        if slot_mapping.dtype != mstype.int32:
            slot_mapping = ops.cast(slot_mapping, mstype.int32)
        cache_write = self.paged_attention_mgr(
            key, value, slot_mapping, batch_valid_length,
            key_cache=key_cache, value_cache=value_cache
        )
        query = ops.depend(query, cache_write)  # write-before-read ordering

        # ---- 2) Gate: should we use FA for this chunk? ----
        use_fa_chunk = (
            self.use_past and
            self.use_flash_attention and
            is_prefill and
            (
                (self.fa_prefill_policy == "all") or
                (self.fa_prefill_policy == "later_only" and (not self.is_first_iteration))
            )
        )

        if use_fa_chunk:
            q_bnsd, k_bnsd, v_bnsd, _ = self._fa_pack_from_cache(
                query, key_cache, value_cache, block_tables, batch_valid_length
            )

            # ensure FP16 end-to-end
            if q_bnsd.dtype != mstype.float16:
                q_bnsd = ops.cast(q_bnsd, mstype.float16)
            if k_bnsd.dtype != mstype.float16:
                k_bnsd = ops.cast(k_bnsd, mstype.float16)
            if v_bnsd.dtype != mstype.float16:
                v_bnsd = ops.cast(v_bnsd, mstype.float16)

            context_layer = self.fa_prefill(
                q_bnsd, k_bnsd, v_bnsd,
                attn_mask, alibi_mask, None, None,
                q_seq_lens, batch_valid_length
            )
        else:
            # fall back to your existing prefill/attention path
            # - for prefill without FA (first chunk when policy="later_only"), use your original core attention
            # - for decode (S_cur == 1), your decode path (paged attention) follows as in repo
            context_layer = self.core_attention(self.layer_index, self.config)(
                query, key, value, attn_mask
            )
            # If your repo calls paged attention for decode here, keep that instead.

        # Mark first prefill as done
        if is_prefill and self.is_first_iteration:
            self.is_first_iteration = False

        # ... post-attention (proj, residual) as in your original code ...
        return context_layer

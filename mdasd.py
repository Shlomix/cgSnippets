# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0

"""Flash Attention Layer (prefill-first & 2D+ prefill-with-gather)."""
__all__ = ['FlashAttention']

from mindspore import Tensor, ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttention(Cell):
    """
    Minimal, robust FlashAttention wrapper for the parallel_core Attention.

    We support:
      • 1st prefill: FA(TH) on (Q,K,V) as provided by the caller.
      • 2nd+ prefill (chunk prefill): gather paged KV cache into a contiguous K,V up to
        total kv length, then FA(TH) with (Q, K_full, V_full).
      • decode: PagedAttention on cached KV.

    Notes:
      • We *derive* actual_seq_{q,kv} from the shapes of the tensors passed to FA to
        avoid dynamic / list-of-tensor pitfalls during the second warm-up.
      • We keep types consistent with cache: if cache exists (float16) we cast q/k/v
        to float16 before touching ReshapeAndCache().
    """

    def __init__(self,
                 head_num,
                 head_dim=None,
                 kv_head_num=None,
                 keep_prob=1.0,
                 scale_value=1.0,
                 pre_tokens=2147483647,
                 next_tokens=2147483647,
                 sparse_mode=0,
                 input_layout="TH",
                 pa_kv_head_num=None,
                 pa_mla_v_dim=0):
        super().__init__()
        self.head_num = head_num
        self.hidden_size_per_attention_head = head_dim
        self.kv_head_num = kv_head_num
        self.sparse_mode = sparse_mode
        self.input_layout = "TH"   # force TH (2-D) for FA
        self.is_prefill = True
        self.use_multi_latent_attention: bool = pa_mla_v_dim > 0

        # cache writer (paged format)
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        # FA kernel (TH only here)
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.input_layout,
            sparse_mode=self.sparse_mode)

        # decode kernel
        self.paged_attention = ops.auto_generate.PagedAttention(
            head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim)

        # tiny helpers
        self._add = ops.Add()
        self._mul = ops.Mul()
        self._reshape = ops.Reshape()
        self._gather = ops.Gather()          # indices int32, axis=0
        self._logical_and = ops.LogicalAnd()
        self._logical_not = ops.LogicalNot()
        self._greater = ops.Greater()
        self._reduce_max = ops.ReduceMax(keep_dims=False)
        self._zeros_like = ops.ZerosLike()
        self._ones_like = ops.OnesLike()
        self._cast = ops.Cast()
        self._shape = ops.Shape()

    # ------------ printing ------------
    def _p(self, tag, x):
        # prints only when layer 5 caller enabled prints; harmless otherwise
        try:
            print(f"{tag}\n{x}")
        except Exception:
            print(f"{tag}: <no-repr>")

    # ------------ detector & gather ------------
    def _is_second_plus_chunk(self, q_seq_lens):
        """Heuristic: we’re in chunk prefill when not first prefill and max(q_seq_lens)>1."""
        if q_seq_lens is None:
            has_chunk_t = Tensor(False, mstype.bool_)
        else:
            # q_seq_lens: Tensor[int32] with shape [B]; we only care if any > 1
            has_chunk_t = self._greater(self._reduce_max(q_seq_lens), Tensor(1, q_seq_lens.dtype))
        # first/second decision is made by caller toggling self.is_prefill
        second_plus = self._logical_and(self._logical_not(Tensor(self.is_prefill, mstype.bool_)),
                                        has_chunk_t)
        return second_plus

    def _gather_contiguous_kv(self, kv_cache, block_tables, total_kv_len):
        """
        Build a contiguous [T_kv, kv_heads*head_dim] view from paged cache for single-batch.
        kv_cache: (num_blocks, block_size, kv_heads, head_dim) or (..., hidden) if NZ packed.
        block_tables: (B, max_blocks) with 1-based ids, 0 padded; here B==1 per partition.
        total_kv_len: python int (total context length for this batch).
        """
        # expect normal layout (num_blocks, block, kv_heads, head_dim)
        num_blocks, block_size, kv_heads, head_dim = self._shape(kv_cache)
        # number of non-zero block ids we need
        needed_blocks = int((total_kv_len + block_size - 1) // block_size)

        # take first needed_blocks entries, convert to 0-based indices
        blocks_1b = block_tables[:, :needed_blocks]      # (1, needed_blocks)
        blocks_0b = self._add(blocks_1b, Tensor(-1, blocks_1b.dtype))
        blocks_0b = self._reshape(blocks_0b, (needed_blocks,))  # (needed_blocks,)

        # gather along block dimension
        gathered = self._gather(kv_cache, blocks_0b, 0)  # (needed_blocks, block, kv_heads, head_dim)

        # trim to exact token length and collapse to TH=(T_kv, H_kv)
        flat = self._reshape(gathered, (needed_blocks * block_size, kv_heads * head_dim))
        flat2d = self._reshape(flat, (int(total_kv_len), kv_heads * head_dim))
        return flat2d

    # ------------ main ------------
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
        """
        Inputs follow the parallel_core Attention caller.
        Prefill: Q/K/V are 2-D (T, H) in TH layout per partition.
        """

        # --- keep dtypes consistent with cache when present ---
        if key_cache is not None and value_cache is not None:
            q = self._cast(query, mstype.float16)
            k = self._cast(key,   mstype.float16)
            v = self._cast(value, mstype.float16)
        else:
            q, k, v = query, key, value

        # If caches exist, first write the current chunk into them.
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(k, v, key_cache, value_cache, slot_mapping)

        # ---- PREP common guards ----
        # Ascend FA (TH) does not accept padding_mask; enforce None.
        padding_mask = None

        # ---- 1) First prefill: FA on (q,k,v) as-is ----
        if self.is_prefill:
            # derive lengths from shapes (compile-time constants), avoid dynamic lists
            tq = (self._shape(q)[0],)      # tuple[int]
            tk = (self._shape(k)[0],)
            _, _, _, context = self.flash_attention(q, k, v,
                                                    None, None,    # no alibi / rel_shift here
                                                    padding_mask, attn_mask,
                                                    prefix, tq, tk)
            self._p("FA context (1st prefill)", context)
            return context

        # ---- 2) 2D+ chunk prefill: gather KV then FA ----
        second_plus = self._is_second_plus_chunk(q_seq_lens)
        can_gather = (key_cache is not None) and (value_cache is not None) and (block_tables is not None)
        use_fa_gather = bool(second_plus.asnumpy().item()) and can_gather

        if use_fa_gather:
            # total kv len comes from the caller (static) or from batch_valid_length
            if isinstance(actual_seq_kvlen, (tuple, list)) and len(actual_seq_kvlen) > 0:
                total_kv_len = int(actual_seq_kvlen[-1])
            else:
                # fallback: batch_valid_length is [B], B==1
                total_kv_len = int(batch_valid_length.asnumpy().reshape(-1)[0])

            k_full = self._gather_contiguous_kv(key_cache, block_tables, total_kv_len)
            v_full = self._gather_contiguous_kv(value_cache, block_tables, total_kv_len)

            # lengths again from shapes of the tensors we actually pass to the kernel
            tq = (self._shape(q)[0],)
            tk = (self._shape(k_full)[0],)

            _, _, _, context = self.flash_attention(q, k_full, v_full,
                                                    None, None,
                                                    padding_mask, attn_mask,
                                                    prefix, tq, tk)
            self._p("FA context (2D+ prefill)", context)
            return context

        # ---- 3) Decode: PagedAttention on cached KV ----
        # (or fallback when gather/FA not applicable)
        if self.use_multi_latent_attention:
            context = self.paged_attention(q, key_cache, key_cache,
                                           block_tables, batch_valid_length, None,
                                           None, attn_mask, q_seq_lens)
        else:
            context = self.paged_attention(q, key_cache, value_cache,
                                           block_tables, batch_valid_length, None,
                                           None, attn_mask, q_seq_lens)
        return context

# ---- mindformers/parallel_core/inference/transformer/flash_attention.py ----
from mindspore import ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore

class FlashAttention(Cell):
    def __init__(self,
                 head_num,
                 head_dim=None,
                 kv_head_num=None,
                 keep_prob=1.0,
                 scale_value=1.0,
                 pre_tokens=2147483647,
                 next_tokens=2147483647,
                 sparse_mode=0,
                 input_layout="BSND",     # <— use BSND for FA
                 pa_kv_head_num=None,
                 pa_mla_v_dim=0):
        super().__init__()
        self.head_num = int(head_num)
        self.hidden_size_per_attention_head = int(head_dim) if head_dim is not None else None
        self.kv_head_num = int(kv_head_num) if kv_head_num is not None else int(head_num)
        self.sparse_mode = int(sparse_mode)
        self.is_prefill = True
        self.use_multi_latent_attention = (pa_mla_v_dim > 0)

        # ops
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.reduce_max = ops.ReduceMax(False)
        self.scalar_to_tensor = ops.ScalarToTensor()
        self.greater = ops.Greater()
        self.gather = ops.Gather()

        # write current chunk to cache each call so PA always works
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

        # FlashAttention in BSND (avoids TND pitfalls)
        self.flash_attention = FlashAttentionScore(
            head_num=self.head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout="BSND",
            sparse_mode=self.sparse_mode
        )

        # Decode
        self.paged_attention = ops.auto_generate.PagedAttention(
            self.head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim
        )

    # ---------------- helpers ----------------
    def _p(self, *msg):
        try:
            print(*msg)
        except Exception:
            pass

    def _bsh_to_bsnd(self, x_bsh, n_heads, hd):
        # [B,S,H] -> [B,S,N,D]
        B, S, H = x_bsh.shape
        assert H == n_heads * hd, f"H={H} must equal N*D={n_heads*hd}"
        x = self.reshape(x_bsh, (B, S, n_heads, hd))
        return x

    def _th_to_bsnd(self, x_th, n_heads, hd):
        # [T,H] -> [1,T,N,D]
        T, H = x_th.shape
        assert H == n_heads * hd, f"H={H} must equal N*D={n_heads*hd}"
        x = self.reshape(x_th, (1, T, n_heads, hd))
        return x

    def _to_bsnd(self, x, n_heads, hd):
        if x.ndim == 3:
            return self._bsh_to_bsnd(x, n_heads, hd)  # [B,S,H]
        elif x.ndim == 2:
            return self._th_to_bsnd(x, n_heads, hd)   # [T,H]
        else:
            raise ValueError(f"Unsupported rank {x.ndim} for tensor in FA")

    def _harmonize_with_cache_dtype(self, q, k, v, key_cache):
        if key_cache is not None:
            dt = key_cache.dtype
            q = self.cast(q, dt); k = self.cast(k, dt); v = self.cast(v, dt)
        return q, k, v

    def _is_second_plus_chunk(self, q_seq_lens):
        if q_seq_lens is None:
            return False
        mx = self.reduce_max(q_seq_lens)
        return bool(self.greater(mx, self.scalar_to_tensor(1, q_seq_lens.dtype)).asnumpy().item())

    def _gather_contiguous_kv(self, kv_cache, block_tables, total_kv_len):
        # kv_cache: [num_blocks, block_size, Nk, D]; return [1, Tk, Nk, D]
        num_blocks, block_size, kv_heads, hd = map(int, kv_cache.shape)
        need = (int(total_kv_len) + block_size - 1) // block_size
        blocks_1b = block_tables[0][:need]
        blocks_0b = blocks_1b - ops.ones_like(blocks_1b)
        gathered = self.gather(kv_cache, blocks_0b, 0)            # [need, block, Nk, D]
        flat = self.reshape(gathered, (-1, kv_heads, hd))          # [need*block, Nk, D]
        flat = flat[:int(total_kv_len)]                            # [Tk, Nk, D]
        return self.reshape(flat, (1, int(total_kv_len), kv_heads, hd))  # [1,Tk,Nk,D]

    # ---------------- main ----------------
    def construct(self,
                  query, key, value,
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

        # Always write this chunk into cache → PA always valid
        if not self.use_multi_latent_attention:
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # Never pass masks to FA (BSND) → avoid AddExt [T,S] vs [T,N,S] mismatch
        if attn_mask is not None:
            self._p("[FA/BSND] dropping attn_mask for FA; was", attn_mask.shape)
        _fa_padding_mask = None
        _fa_attn_mask = None

        Nq = self.head_num
        Nk = self.kv_head_num
        D  = self.hidden_size_per_attention_head

        # 1) First prefill → FA(BSND) on current chunk
        if self.is_prefill:
            q_bsnd = self._to_bsnd(query, Nq, D)   # [B,Sq,Nq,D]
            k_bsnd = self._to_bsnd(key,   Nk, D)   # [B,Sk,Nk,D]
            v_bsnd = self._to_bsnd(value, Nk, D)
            q_bsnd, k_bsnd, v_bsnd = self._harmonize_with_cache_dtype(q_bsnd, k_bsnd, v_bsnd, key_cache)
            # (q,k,v, alibi, rel_shift, padding_mask, attn_mask, prefix, actual_q, actual_kv)
            _, _, _, ctx = self.flash_attention(q_bsnd, k_bsnd, v_bsnd,
                                                None, None,
                                                _fa_padding_mask, _fa_attn_mask,
                                                prefix, actual_seq_qlen, actual_seq_kvlen)
            # self._p("[FA/BSND] 1st prefill:", q_bsnd.shape, k_bsnd.shape, v_bsnd.shape, "->", ctx.shape)
            return ctx

        # 2) 2nd+ prefill → gather contiguous KV from cache, FA(BSND)
        if self._is_second_plus_chunk(q_seq_lens) and (key_cache is not None) and (value_cache is not None) \
           and (block_tables is not None) and (actual_seq_kvlen is not None):
            total_kv_len = int(actual_seq_kvlen.asnumpy()[-1])
            k_full = self._gather_contiguous_kv(key_cache,   block_tables, total_kv_len)  # [1,Tk,Nk,D]
            v_full = self._gather_contiguous_kv(value_cache, block_tables, total_kv_len)  # [1,Tk,Nk,D]
            q_bsnd = self._to_bsnd(query, Nq, D)  # [B(=1),Tq,Nq,D] or [B,Sq,Nq,D]

            q_bsnd, k_full, v_full = self._harmonize_with_cache_dtype(q_bsnd, k_full, v_full, key_cache)

            _, _, _, ctx = self.flash_attention(q_bsnd, k_full, v_full,
                                                None, None,
                                                _fa_padding_mask, _fa_attn_mask,
                                                prefix, actual_seq_qlen, actual_seq_kvlen)
            # self._p("[FA/BSND] 2D+ prefill:", q_bsnd.shape, k_full.shape, v_full.shape, "->", ctx.shape)
            return ctx

        # 3) Decode / fallback → PagedAttention (keeps attn_mask)
        if self.use_multi_latent_attention:
            ctx = self.paged_attention(query, key_cache, key_cache,
                                       block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)
        else:
            if key_cache is not None:
                query = self.cast(query, key_cache.dtype)
            ctx = self.paged_attention(query, key_cache, value_cache,
                                       block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)
        return ctx
# ---- end file ----

# mindformers/modules/flash_attention.py  (replace the whole FlashAttention class with this)

import os
from mindspore import ops, dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops.operations.nn_ops import FlashAttentionScore

__all__ = ['FlashAttention']


class FlashAttention(Cell):
    """
    BSH FlashAttention + PagedAttention bridge.

    Behavior
      • First prefill (no prior ctx): FlashAttention (BSH).
      • Second+ prefill (has prior ctx): FlashAttention (BSH) over full prefix by building dense KV from paged cache
        with **blockwise gather** and **clamped length** (batch=1 PoC). If constraints not met, fallback to PagedAttention
        (unless MF_MANIFEST_FA_MISMATCH=1).
      • Decode: PagedAttention.

    Env toggles
      MF_DEBUG_ATTENTION=1        -> print key steps.
      MF_FORCE_FA_CHUNK=1         -> (default=1) attempt FA on 2nd+ prefill (else PA).
      MF_MANIFEST_FA_MISMATCH=1   -> force FA even if not x16 or B>1 (to demo kernel failure).
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
                 input_layout="BSH",
                 pa_kv_head_num=None,
                 pa_mla_v_dim=0):
        super().__init__()
        if input_layout != "BSH":
            raise ValueError("FlashAttention wrapper expects input_layout='BSH' in this branch.")

        # Model params
        self.head_num = int(head_num)
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.kv_head_num = int(kv_head_num) if kv_head_num is not None else None
        self.sparse_mode = sparse_mode
        self.is_prefill = True
        self.use_multi_latent_attention: bool = pa_mla_v_dim > 0

        # Env toggles
        self.debug = os.getenv("MF_DEBUG_ATTENTION", "0") == "1"
        self.force_fa_chunk = os.getenv("MF_FORCE_FA_CHUNK", "1") == "1"
        self.manifest_mismatch = os.getenv("MF_MANIFEST_FA_MISMATCH", "0") == "1"

        # Common ops
        self.Cast = ops.Cast()
        self.Reshape = ops.Reshape()
        self.Print = ops.Print()
        self.Range = ops.Range()
        self.ExpandDims = ops.ExpandDims()
        self.Mod = ops.Mod()
        self.ReduceSum = ops.ReduceSum(keep_dims=False)
        self.ReduceMax = ops.ReduceMax(keep_dims=False)
        self.Gather = ops.Gather()
        self.FloorDiv = ops.FloorDiv()
        self.Minimum = ops.Minimum()
        self.Maximum = ops.Maximum()
        self.ConcatH = ops.Concat(axis=1)  # head axis in [T, heads, dim]
        self.BroadcastTo = ops.BroadcastTo

        # Kernels
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout="BSH",
            sparse_mode=self.sparse_mode
        )
        self.paged_attention = ops.auto_generate.PagedAttention(
            head_num, scale_value, pa_kv_head_num, mla_v_dim=pa_mla_v_dim
        )
        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()

    # ---------------- debug helper ----------------
    def _dbg(self, *args):
        if self.debug:
            self.Print(*args)

    # ---------------- small helpers ----------------
    def _all_x16(self, x_i32):
        rem = self.Mod(x_i32, ops.scalar_to_tensor(16, mstype.int32))
        s = self.ReduceSum(rem, 0)
        return s == ops.scalar_to_tensor(0, mstype.int32)

    def _repeat_kv_to_q_heads(self, dense_kv):
        """[T_kv, kv_heads*head_dim] -> [T_kv, head_num*head_dim] if kv_heads < head_num."""
        if self.kv_head_num is None or self.head_dim is None:
            return dense_kv
        if self.kv_head_num == self.head_num:
            return dense_kv
        rep = self.head_num // self.kv_head_num
        x = self.Reshape(dense_kv, (-1, self.kv_head_num, self.head_dim))
        x = self.ConcatH(tuple([x] * rep))              # concat along head axis
        x = self.Reshape(x, (-1, self.head_num * self.head_dim))
        return x

    def _sanitize_fa_inputs(self, q_bsh, k_bsh, attn_mask, aql, akl):
        """For FA: cast attn_mask to q.dtype and ensure shapes [B,Sq,Sk]. Lengths int32."""
        B, Sq, _ = q_bsh.shape
        Sk = int(k_bsh.shape[1])
        qdtype = q_bsh.dtype

        # attn_mask -> [B,Sq,Sk]
        if attn_mask is None:
            attn_mask = ops.zeros((B, Sq, Sk), qdtype)
        else:
            am = attn_mask
            if len(am.shape) == 2:
                am = self.ExpandDims(am, 0)            # [1,Sq,Sk]
            if am.shape[0] == 1 and B > 1:
                am = self.BroadcastTo((B, am.shape[1], am.shape[2]))(am)
            if am.dtype != qdtype:
                am = self.Cast(am, qdtype)
            attn_mask = am

        if aql is None:
            aql = ops.fill(mstype.int32, (B,), Sq)
        elif aql.dtype != mstype.int32:
            aql = self.Cast(aql, mstype.int32)

        if akl is None:
            akl = ops.fill(mstype.int32, (B,), Sk)     # derive from real Sk
        elif akl.dtype != mstype.int32:
            akl = self.Cast(akl, mstype.int32)

        return attn_mask, aql, akl

    def _dense_kv_from_cache_b1(self, key_cache, value_cache, block_tables, kv_end_req_i32):
        """
        Build dense KV contiguously from paged cache for B=1 using **blockwise gather** with **clamped length**.

        key_cache:  [num_blocks, block_size, kv_heads*head_dim] or [num_blocks, block_size, kv_heads, head_dim]
        value_cache: same layout and dtype as key_cache
        block_tables: [1, num_blocks] int32 with physical block ids (unused tail may be -1)
        kv_end_req_i32: scalar int32 requested prefix length (prev_ctx + chunk)

        Returns: (k_dense, v_dense, block_size)
                 where k_dense/v_dense are TH: [T_kv, H_kv] with T_kv = min(kv_end_req, num_blocks*block_size)
        """
        bt = block_tables[0]                                  # [num_blocks]
        shape = key_cache.shape
        if len(shape) == 4:
            nb, bs, kvh, hd = shape
            H_kv = kvh * hd
            kc_nbbsH = self.Reshape(key_cache, (nb, bs, H_kv))
            vc_nbbsH = self.Reshape(value_cache, (nb, bs, H_kv))
        elif len(shape) == 3:
            nb, bs, H_kv = shape
            kc_nbbsH = key_cache
            vc_nbbsH = value_cache
        else:
            nb, bs, H_kv = int(shape[0]), int(shape[1]), int(shape[-1])
            kc_nbbsH = self.Reshape(key_cache, (nb, bs, H_kv))
            vc_nbbsH = self.Reshape(value_cache, (nb, bs, H_kv))

        # Clamp kv_end to real capacity nb*bs
        nb_t = ops.scalar_to_tensor(nb, mstype.int32)
        bs_t = ops.scalar_to_tensor(bs, mstype.int32)
        cap = nb_t * bs_t
        kv_end = self.Minimum(kv_end_req_i32, cap)            # scalar int32

        # used_blocks = ceil(kv_end / bs), but never exceed nb
        used_blocks = self.FloorDiv(kv_end + (bs_t - 1), bs_t)
        used_blocks = self.Minimum(used_blocks, nb_t)

        # physical block ids for first used blocks; clamp [-1, nb-1] → [0, nb-1]
        rng = self.Range(ops.scalar_to_tensor(0, mstype.int32), used_blocks, ops.scalar_to_tensor(1, mstype.int32))
        ids = self.Gather(bt, rng, 0)                         # [used_blocks]
        zero = ops.scalar_to_tensor(0, mstype.int32)
        nbm1 = ops.scalar_to_tensor(nb - 1, mstype.int32)
        ids = self.Maximum(ids, zero)
        ids = self.Minimum(ids, nbm1)

        # blockwise gather → [used_blocks, bs, H_kv]
        kc_used = self.Gather(kc_nbbsH, ids, 0)
        vc_used = self.Gather(vc_nbbsH, ids, 0)

        # flatten to [used_blocks*bs, H_kv] and slice first kv_end tokens (contiguous)
        flat_k = self.Reshape(kc_used, (-1, H_kv))
        flat_v = self.Reshape(vc_used, (-1, H_kv))
        idx0 = self.Range(zero, kv_end, ops.scalar_to_tensor(1, mstype.int32))
        k_dense = self.Gather(flat_k, idx0, 0)                # [T_kv, H_kv]
        v_dense = self.Gather(flat_v, idx0, 0)
        return k_dense, v_dense, bs

    def _run_paged(self, query, key_cache, value_cache, block_tables, batch_valid_length, q_seq_lens, attn_mask):
        want = (key_cache.dtype if key_cache is not None else query.dtype)
        if attn_mask is not None and attn_mask.dtype != want:
            attn_mask = self.Cast(attn_mask, want)
        if self.use_multi_latent_attention:
            return self.paged_attention(query, key_cache, key_cache,
                                        block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)
        return self.paged_attention(query, key_cache, value_cache,
                                    block_tables, batch_valid_length, None, None, attn_mask, q_seq_lens)

    # ---------------- forward ----------------
    def construct(self,
                  query,            # [B,Sq,H]
                  key,              # [B,Sq,H_kv]
                  value,            # [B,Sq,H_kv]
                  slot_mapping=None,
                  block_tables=None,
                  batch_valid_length=None,  # [B] context length BEFORE this chunk
                  context_lens_tensor=None,
                  q_seq_lens=None,          # [B]
                  actual_seq_qlen=None,     # [B]
                  actual_seq_kvlen=None,    # [B]
                  attn_mask=None,           # [B,Sq,Sk] or [Sq,Sk]
                  padding_mask=None,        # MUST be None for FA
                  prefix=None,
                  key_cache=None,
                  value_cache=None):
        """Forward process."""
        B, Sq, _ = query.shape
        caches_exist = (key_cache is not None) and (value_cache is not None)
        target_dtype = key_cache.dtype if caches_exist else query.dtype

        # align dtypes
        if query.dtype != target_dtype:
            query = self.Cast(query, target_dtype)
        if key is not None and key.dtype != target_dtype:
            key = self.Cast(key, target_dtype)
        if value is not None and value.dtype != target_dtype:
            value = self.Cast(value, target_dtype)

        # write current chunk into cache (so kv_end includes the chunk)
        if caches_exist and (slot_mapping is not None) and (not self.use_multi_latent_attention):
            if slot_mapping.dtype != mstype.int32:
                slot_mapping = self.Cast(slot_mapping, mstype.int32)
            self.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

        # ---------------- PREFILL ----------------
        if self.is_prefill:
            padding_mask = None  # Ascend FA requirement

            has_prior_ctx = (batch_valid_length is not None) and \
                            (int(self.ReduceMax(self.Cast(batch_valid_length, mstype.int32)).asnumpy()) > 0)

            # Prefill-1: FA on current chunk tensors
            if not has_prior_ctx:
                aql = actual_seq_qlen if actual_seq_qlen is not None else ops.fill(mstype.int32, (B,), Sq)
                akl = actual_seq_kvlen if actual_seq_kvlen is not None else ops.fill(mstype.int32, (B,), Sq)
                aql = self.Cast(aql, mstype.int32)
                akl = self.Cast(akl, mstype.int32)

                aligned = self._all_x16(aql) and self._all_x16(akl)
                if not aligned and not self.manifest_mismatch:
                    self._dbg("Prefill-1: not x16 -> PA fallback; aql=", aql, " akl=", akl)
                    return self._run_paged(query, key_cache, value_cache, block_tables,
                                           batch_valid_length, q_seq_lens, attn_mask)

                am, aql, akl = self._sanitize_fa_inputs(query, key, attn_mask, aql, akl)
                self._dbg("Prefill-1: FA; aql=", aql, " akl=", akl, " dtype=", query.dtype)
                _, _, _, ctx = self.flash_attention(query, key, value,
                                                    None, None,  # real_shift, alibi
                                                    None,        # padding_mask MUST be None
                                                    am,          # attn_mask
                                                    None,        # prefix
                                                    aql, akl)
                return ctx

            # Prefill-2+: FA over full prefix (batch=1 PoC) or fallback to PA
            if not self.force_fa_chunk:
                self._dbg("Prefill-2+: forced off -> PA")
                return self._run_paged(query, key_cache, value_cache, block_tables,
                                       batch_valid_length, q_seq_lens, attn_mask)

            if B != 1 and not self.manifest_mismatch:
                self._dbg("Prefill-2+: B!=1 -> PA fallback")
                return self._run_paged(query, key_cache, value_cache, block_tables,
                                       batch_valid_length, q_seq_lens, attn_mask)

            # q and kv lengths (int32)
            if q_seq_lens is None:
                q_seq_lens = ops.fill(mstype.int32, (B,), Sq)
            q_seq_lens = self.Cast(q_seq_lens, mstype.int32)
            bvl = self.Cast(batch_valid_length, mstype.int32)
            kv_end_req = (bvl + q_seq_lens)[0]  # scalar int32 for B=1 request

            # build dense KV with **clamped length** (prevents OOB)
            bt_i32 = self.Cast(block_tables, mstype.int32)
            k_dense, v_dense, block_size = self._dense_kv_from_cache_b1(
                key_cache, value_cache, bt_i32, kv_end_req
            )  # [T_kv, H_kv] with T_kv = min(kv_end_req, nb*bs)

            # heads repeat if needed, then reshape to BSH
            k_dense = self._repeat_kv_to_q_heads(k_dense)       # [T_kv, H]
            v_dense = self._repeat_kv_to_q_heads(v_dense)
            T_kv = int(k_dense.shape[0])
            k_bsh = self.Reshape(k_dense, (1, T_kv, -1))        # [1,Sk,H]
            v_bsh = self.Reshape(v_dense, (1, T_kv, -1))

            # x16 alignment check **after** we know real Sk (T_kv)
            aligned = self._all_x16(q_seq_lens) and (T_kv % 16 == 0)
            if (not aligned) and (not self.manifest_mismatch):
                self._dbg("Prefill-2+: not x16 -> PA fallback; q=", q_seq_lens, " Sk(T_kv)=", T_kv)
                return self._run_paged(query, key_cache, value_cache, block_tables,
                                       batch_valid_length, q_seq_lens, attn_mask)

            # sanitize inputs for FA; let akl be derived from k_bsh.shape
            am, aql, akl = self._sanitize_fa_inputs(query, k_bsh, attn_mask, q_seq_lens, None)
            self._dbg("Prefill-2+: FA(gathered+clamped); aql=", aql, " akl=", akl,
                      " dtype=", query.dtype, " blk_sz=", block_size)

            _, _, _, ctx = self.flash_attention(query, k_bsh, v_bsh,
                                                None, None,
                                                None,   # padding_mask MUST be None
                                                am,     # attn_mask
                                                None,   # prefix
                                                aql, akl)
            return ctx

        # ---------------- DECODE / PA ----------------
        self._dbg("Decode/PA; dtype=", query.dtype)
        return self._run_paged(query, key_cache, value_cache, block_tables,
                               batch_valid_length, q_seq_lens, attn_mask)

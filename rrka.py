# ============================
# 1) Inside class Attention.__init__(...) add this line (after other ops inits are OK):
# ============================
self._print_op = ops.Print()  # single Print op for this layer


# ============================
# 2) Also inside class Attention (define right after __init__), add this helper:
# ============================
def _print(self, *args):
    """Conditional print: only fires for layer 5; graph-safe (ops.Print)."""
    if self.layer_number == 5:
        self._print_op(*args)


# ============================
# 3) In Attention.construct(...), REPLACE your previous classification/prints block
#    (the one that used .asnumpy()/bool(...)) with the following graph-safe code.
#    Put it right after rotary embedding and BEFORE calling self.core_attention.
# ============================
from mindspore import dtype as mstype

# tiny local aliases
_reduce_sum  = ops.ReduceSum(keep_dims=False)
_reduce_max  = ops.ReduceMax(keep_dims=False)
_equal       = ops.Equal()
_greater     = ops.Greater()
_logical_and = ops.LogicalAnd()
_logical_not = ops.LogicalNot()
_fill        = ops.Fill()
_cast        = ops.Cast()
_scalar_bool = lambda b: ops.scalar_to_tensor(bool(b), mstype.bool_)

# Build tensor booleans ONLY; never convert to Python bool here.
# Decode: all q_seq_lens == 1  →  reduce_all(q_seq_lens == 1)
if q_seq_lens is not None:
    ones_like_q = _fill(q_seq_lens.dtype, q_seq_lens.shape, 1)
    decode_vec  = _equal(q_seq_lens, ones_like_q)            # [B] bool
    is_decode_t = ops.ReduceAll(keep_dims=False)(decode_vec)  # scalar bool
else:
    is_decode_t = _scalar_bool(False)

# max lengths (fall back to 0 if None so comparisons stay defined)
if actual_seq_qlen is not None:
    max_q = _reduce_max(actual_seq_qlen)   # scalar (int)
else:
    max_q = ops.scalar_to_tensor(0, mstype.int32)
if actual_seq_kvlen is not None:
    max_kv = _reduce_max(actual_seq_kvlen) # scalar (int)
else:
    max_kv = ops.scalar_to_tensor(0, mstype.int32)

kv_longer_than_q_t = _greater(max_kv, max_q)                 # bool
same_len_t         = _equal(max_kv, max_q)                   # bool
is_prefill_t       = _scalar_bool(self.is_prefill)           # wrap python flag as tensor bool

# first prefill: self.is_prefill && (kv_len == q_len)
first_prefill_t    = _logical_and(is_prefill_t, same_len_t)

# second+ chunk: (!is_prefill) && (!decode) && (kv_len > q_len)
second_plus_chunk_t = _logical_and(
    _logical_not(is_prefill_t),
    _logical_and(_logical_not(is_decode_t), kv_longer_than_q_t)
)

# Lightweight, layer-5-only prints (these accept tensors directly)
self._print("ATTN.class layer=5",
            " is_prefill=", is_prefill_t,
            " is_decode=",  is_decode_t,
            " first_prefill=", first_prefill_t,
            " second_plus_chunk=", second_plus_chunk_t,
            " max_q=", max_q, " max_kv=", max_kv)

# (No Python branching on these flags here; let the downstream module decide control flow.
#  If you must branch, use tensor-based control or move the decision into the FA module,
#  which already handles the chunk2+ path.)

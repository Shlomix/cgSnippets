# inside Attention.construct() where you currently print the flags:
_ReduceMax  = ops.ReduceMax(keep_dims=False)
_Greater    = ops.Greater()
_LogicalAnd = ops.LogicalAnd()
_LogicalNot = ops.LogicalNot()
_scalar_bool = lambda b: ops.scalar_to_tensor(bool(b), mstype.bool_)

is_prefill_t = _scalar_bool(self.is_prefill)
if q_seq_lens is not None:
    has_chunk_t = _Greater(_ReduceMax(q_seq_lens), ops.scalar_to_tensor(1, q_seq_lens.dtype))
else:
    has_chunk_t = _scalar_bool(False)

first_prefill_t     = is_prefill_t
second_plus_chunk_t = _LogicalAnd(_LogicalNot(is_prefill_t), has_chunk_t)

self._print("ATTN.class layer=5",
            " is_prefill=", is_prefill_t,
            " has_chunk=", has_chunk_t,
            " second_plus_chunk=", second_plus_chunk_t)

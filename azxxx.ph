# --- in mindformers/modules/flash_attention.py (inside class FlashAttention) ---
from mindspore import dtype as mstype
import mindspore.ops as ops

def _classify_step(self, q_seq_lens, actual_seq_qlen, actual_seq_kvlen, batch_valid_length, bsz):
    """Simplified, robust classifier.
       second_plus_chunk := (!is_prefill) && any(q_len > 1)
       first_prefill     := is_prefill
       is_decode_t       := !any(q_len > 1)  (for info/logging only)"""
    _ReduceMax  = ops.ReduceMax(keep_dims=False)
    _Greater    = ops.Greater()
    _LogicalAnd = ops.LogicalAnd()
    _LogicalNot = ops.LogicalNot()

    # tensor bool wrappers
    is_prefill_t = ops.scalar_to_tensor(bool(self.is_prefill), mstype.bool_)

    if q_seq_lens is not None:
        # has_chunk_t = any(q_len > 1)
        one = ops.scalar_to_tensor(1, q_seq_lens.dtype)
        has_chunk_t = _Greater(_ReduceMax(q_seq_lens), one)
    else:
        has_chunk_t = ops.scalar_to_tensor(False, mstype.bool_)

    first_prefill_t      = is_prefill_t
    second_plus_chunk_t  = _LogicalAnd(_LogicalNot(is_prefill_t), has_chunk_t)
    is_decode_t          = _LogicalNot(has_chunk_t)  # informational

    self._maybe_print("FA.cls is_prefill=", is_prefill_t,
                      " has_chunk=", has_chunk_t,
                      " second_plus_chunk=", second_plus_chunk_t)

    return is_decode_t, first_prefill_t, second_plus_chunk_t

# ==== STEP 2 (JIT-safe, concise) ====
from mindspore import ops as P, Tensor, dtype as mstype
from mindspore.ops import functional as F

_shape = P.Shape()
_reduce_max = P.ReduceMax(keep_dims=False)
_greater = P.Greater()
_select = P.Select()
_pr = P.Print()

# Current sequence length S (safe even if rank==1)
S_val = _shape(input_ids)[-1]           # python int known at compile-time
S = Tensor(S_val, mstype.int32)         # tensor scalar for comparisons/print

# is_prefill: S > 1
is_prefill = _greater(S, Tensor(1, mstype.int32))

# has_past: max(batch_valid_length) > 0  (or False if None)
if batch_valid_length is not None:
    bvl_max = _reduce_max(batch_valid_length)       # (scalar tensor)
    has_past = _greater(bvl_max, Tensor(0, mstype.int32))
else:
    has_past = Tensor(False, mstype.bool_)

# phase_code: 0 = decode, 1 = first_prefill, 2 = later_prefill
phase_code = _select(
    is_prefill,
    _select(has_past, Tensor(2, mstype.int32), Tensor(1, mstype.int32)),
    Tensor(0, mstype.int32)
)

# Print + force execution
input_ids = F.depend(input_ids, _pr("[STEP2][qwen2_5.construct] phase_code (0=decode,1=first,2=later):"))
input_ids = F.depend(input_ids, _pr(phase_code))
input_ids = F.depend(input_ids, _pr("[STEP2] S:"))
input_ids = F.depend(input_ids, _pr(S))
if q_seq_lens is not None:
    input_ids = F.depend(input_ids, _pr("[STEP2] q_seq_lens:"));         input_ids = F.depend(input_ids, _pr(q_seq_lens))
if batch_valid_length is not None:
    input_ids = F.depend(input_ids, _pr("[STEP2] batch_valid_length:")); input_ids = F.depend(input_ids, _pr(batch_valid_length))
# ==== /STEP 2 ====

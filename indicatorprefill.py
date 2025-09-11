# --- Insert this just after self.reshape_and_cache(...) and before the if self.is_prefill: ---
from mindspore import ops

print_op = ops.Print()
reduce_sum = ops.ReduceSum(keep_dims=False)
reduce_max = ops.ReduceMax(keep_dims=False)
greater = ops.Greater()
equal = ops.Equal()

# Batch size: prefer query.shape[0] if available here; otherwise fall back to batch_valid_length
try:
    bsz = query.shape[0]
except Exception:
    bsz = batch_valid_length.shape[0] if batch_valid_length is not None else -1

# Safe summaries (no exceptions)
if q_seq_lens is None:
    print_op("CHK q_seq_lens=None")
else:
    print_op("CHK q_seq_lens shape=", str(q_seq_lens.shape), " dtype=", str(q_seq_lens.dtype), " val=", str(q_seq_lens))
if actual_seq_qlen is None:
    print_op("CHK actual_seq_qlen=None")
else:
    print_op("CHK actual_seq_qlen shape=", str(actual_seq_qlen.shape), " dtype=", str(actual_seq_qlen.dtype), " val=", str(actual_seq_qlen))
if actual_seq_kvlen is None:
    print_op("CHK actual_seq_kvlen=None")
else:
    print_op("CHK actual_seq_kvlen shape=", str(actual_seq_kvlen.shape), " dtype=", str(actual_seq_kvlen.dtype), " val=", str(actual_seq_kvlen))

# --- Classify step (decode vs first prefill vs 2nd+ chunk prefill) ---
# decode: q_seq_lens exists and all ones → sum(q_seq_lens) == batch_size
# 2nd+ chunk: not first prefill, q_seq_lens exists, sum(q_seq_lens) > batch_size, and kv prefix longer than q → max(kv) > max(q)

is_decode = False
is_second_plus_chunk = False
is_first_prefill = False

if q_seq_lens is not None:
    sum_q = reduce_sum(q_seq_lens)   # scalar tensor
    # compare sum(q_seq_lens) to batch size
    if bsz != -1:
        # build a scalar tensor for comparison with same dtype as sum_q
        bsz_tensor = ops.scalar_to_tensor(bsz, sum_q.dtype)
        is_decode = bool(equal(sum_q, bsz_tensor).asnumpy())  # True if all q_lens==1

if (actual_seq_qlen is not None) and (actual_seq_kvlen is not None):
    max_q = reduce_max(actual_seq_qlen)
    max_kv = reduce_max(actual_seq_kvlen)
    kv_longer_than_q = bool(greater(max_kv, max_q).asnumpy())
    same_len = bool(equal(max_kv, max_q).asnumpy())
else:
    kv_longer_than_q = False
    same_len = False

# In many stacks, first prefill sets self.is_prefill True and has same_len=True
is_first_prefill = bool(self.is_prefill) and same_len

# 2nd+ chunk heuristic (robust to mixed batches):
# - not first prefill
# - we’re not in pure decode (some q>1 this step)
# - kv prefix is longer than this step’s q
is_second_plus_chunk = (not bool(self.is_prefill)) and (not is_decode) and kv_longer_than_q

print_op("CLASS:",
         " is_prefill=", str(self.is_prefill),
         " is_decode=", str(is_decode),
         " first_prefill=", str(is_first_prefill),
         " second_plus_chunk=", str(is_second_plus_chunk))

# If you want to force FlashAttention for 2nd+ chunk here, replace the upcoming
#   if self.is_prefill: ...
#   else: (paged)
# with:
#
# if is_first_prefill:
#     ... call self.flash_attention(...)
# elif is_second_plus_chunk:
#     ... call self.flash_attention(...)   # <= the PoC case
# else:
#     ... call self.paged_attention(...)
#
# Otherwise, keep your existing control flow and use prints to verify.

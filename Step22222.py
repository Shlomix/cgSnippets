# ==== STEP 2: simple probe (no ops, no try) ====
# Works because Tensor.shape is a Python tuple known at compile time.
sh = getattr(input_ids, "shape", None)

# derive current seq length S without indexing [1]
S = -1
if sh and len(sh) >= 1:
    last_dim = sh[-1]            # safe even if rank==1
    S = -1 if last_dim is None else int(last_dim)

prefill = (S > 1)                         # S>1 => prefill call, S==1 => decode step
chunked = (q_seq_lens is not None)        # presence hints chunked prefill
# we can't inspect tensor *values* here, so use a heuristic for "later prefill":
later_prefill = (prefill and chunked and (batch_valid_length is not None))

# light, human-readable summary
bvl_shape = getattr(batch_valid_length, "shape", None)
print(
    f"[STEP2][qwen2_5.construct] phase="
    f"{'decode' if not prefill else ('prefill_chunked' if chunked else 'prefill_full')}"
    f"  S={S}  q_seq_lens={'set' if q_seq_lens is not None else 'None'}"
    f"  bvl_shape={tuple(bvl_shape) if bvl_shape else None}"
    f"  later_prefill_heuristic={'yes' if later_prefill else 'no'}",
    flush=True
)
# ==== /STEP 2 ====

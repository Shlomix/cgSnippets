# ==== STEP 2: simple probe (no shape[1]) ====
from mindspore import ops as P

phase = "decode"
S = 1  # we'll print a best-effort "chunk size"

# If q_seq_lens is provided, we're in (chunked) prefill
if q_seq_lens is not None:
    # chunk length for display
    try:
        S = int(q_seq_lens.asnumpy().max())
    except Exception:
        # fallback if asnumpy is not allowed here
        S = -1

    # batch_valid_length > 0 means we've already cached a prefix -> later prefill
    has_past = False
    if batch_valid_length is not None:
        try:
            bvl_max = int(P.ReduceMax()(batch_valid_length).asnumpy().item())
            has_past = bvl_max > 0
        except Exception:
            pass
    phase = "later_prefill" if has_past else "first_prefill"

else:
    # No q_seq_lens: it's either full prefill (single big call) or decode.
    # Try to infer S without indexing shape[1].
    shp = getattr(input_ids, "shape", None)
    if shp is not None:
        try:
            # last dim is safe to take even if rank==1 (won't crash)
            S = int(shp[-1])
        except Exception:
            S = 1
    phase = "first_prefill" if S > 1 else "decode"

print(f"[STEP2][qwen2_5.construct] phase={phase} S={S}", flush=True)
if q_seq_lens is not None:
    print("[STEP2] q_seq_lens:", q_seq_lens, flush=True)
if batch_valid_length is not None:
    print("[STEP2] batch_valid_length:", batch_valid_length, flush=True)
# ==== /STEP 2 ====

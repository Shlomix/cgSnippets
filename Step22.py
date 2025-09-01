from mindspore import ops as P

S = int(input_ids.shape[1])  # current seq len for this call

phase = "decode"
if S > 1:
    if batch_valid_length is None:
        phase = "first_prefill"
    else:
        # one-liner to see if any past exists (=> second+ prefill chunk)
        bvl_max = int(P.ReduceMax()(batch_valid_length).asnumpy().item())
        phase = "later_prefill" if bvl_max > 0 else "first_prefill"

print(f"[STEP2][qwen2_5.construct] phase={phase} S={S}", flush=True)
if q_seq_lens is not None:
    print("[STEP2] q_seq_lens:", q_seq_lens, flush=True)
if batch_valid_length is not None:
    print("[STEP2] batch_valid_length:", batch_valid_length, flush=True)

# keep a tiny flag you can read later from attention (optional)
self._later_prefill = (phase == "later_prefill")
# ==== /STEP 2 ====

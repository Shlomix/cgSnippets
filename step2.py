
try:
    B = int(input_ids.shape[0])
    S = int(input_ids.shape[1])          # current call's seq len
except Exception:
    B = -1; S = -1

is_prefill = (S > 1)
has_past = False
if batch_valid_length is not None:
    try:
        # bvl is (B,), >0 means we already have prefix cached → 2nd+ chunk
        has_past = bool((batch_valid_length.asnumpy() > 0).any())
    except Exception:
        pass

phase = "decode"
if is_prefill and not has_past:
    phase = "first_prefill"
elif is_prefill and has_past:
    phase = "later_prefill"

print(f"[STEP2][qwen2_5.construct] phase={phase}  B={B} S={S} "
      f"q_seq_lens={None if q_seq_lens is None else list(q_seq_lens.asnumpy())} "
      f"bvl={None if batch_valid_length is None else list(batch_valid_length.asnumpy())}",
      flush=True)

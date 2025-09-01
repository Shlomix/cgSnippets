# ==== STEP 2 (minimal, jit-friendly enough) ====
sh = getattr(input_ids, "shape", None)            # tuple known at compile time
S  = int(sh[-1]) if (sh and len(sh) >= 1 and sh[-1] is not None) else -1

prefill  = (S > 1)                                # S>1 means prefill call (not decode)
chunked  = (q_seq_lens is not None)               # q_seq_lens present => chunked prefill path
bvl_set  = (batch_valid_length is not None)       # we won't read its values here

print(f"[STEP2][qwen2_5.construct] "
      f"S={S}  phase={'prefill' if prefill else 'decode'}  "
      f"q_seq_lens={'set' if chunked else 'None'}  "
      f"batch_valid_length={'set' if bvl_set else 'None'}",
      flush=True)
# ==== /STEP 2 ====

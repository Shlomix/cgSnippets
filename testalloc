#!/usr/bin/env python3
"""
Plain-Python tests for lazy_page_allocator.py (no pytest).

Covers:
  1. Prefill only
  2. Prefill + decode with 1 remote
  3. Prefill + decode with 2 remotes
  4. Capacity check (forces 2nd remote)
"""
from lazy_page_allocator import LazyPageAllocator

def make_sequences(n_seq, tok_per): return [list(range(tok_per)) for _ in range(n_seq)]
def decode_steps(a, steps=4):       [a.step() for _ in range(steps)]

def run():
    combos = [(tp, pp, rem) for tp in (1,2) for pp in (1,2) for rem in (0,1,2)]
    fails  = []
    # --- Test 1 ---------------------------------------------------------
    for tp,pp,rem in combos:
        a = LazyPageAllocator(make_sequences(4,16), tp, pp, rem,
                              warmed_up=True,
                              remote_max_tables=(2 if rem==1 else 1))
        if a.used_remote_worker_ids():
            fails.append(f"PrefillOnly tp={tp} pp={pp} rem={rem}")
    # --- Test 2 ---------------------------------------------------------
    for tp,pp in ((1,1),(2,1),(1,2),(2,2)):
        a = LazyPageAllocator(make_sequences(4,16), tp, pp, remotes=1,
                              warmed_up=True, remote_max_tables=2)
        decode_steps(a)
        if a.used_remote_worker_ids() != [tp*pp]:
            fails.append(f"OneRemote tp={tp} pp={pp}")
    # --- Test 3 ---------------------------------------------------------
    for tp,pp in ((1,1),(2,1),(1,2),(2,2)):
        a = LazyPageAllocator(make_sequences(4,16), tp, pp, remotes=2,
                              warmed_up=True, remote_max_tables=1)
        decode_steps(a)
        base = tp*pp
        if set(a.used_remote_worker_ids()) != {base, base+1}:
            fails.append(f"TwoRemotes tp={tp} pp={pp}")
    # --- Test 4 ---------------------------------------------------------
    for tp,pp in ((1,1),(2,1),(1,2),(2,2)):
        a = LazyPageAllocator(make_sequences(4,16), tp, pp, remotes=2,
                              warmed_up=True, remote_max_tables=1)
        if a.used_remote_worker_ids(): fails.append(f"CapCheck-pre tp={tp} pp={pp}")
        decode_steps(a)
        if len(a.used_remote_worker_ids()) != 2:
            fails.append(f"CapCheck-post tp={tp} pp={pp}")

    # summary
    if fails:
        print("❌ Failures:")
        for f in fails: print("  •", f)
    else:
        print(f"✅ All {len(combos)+3*4} cases passed!")

if __name__ == "__main__":
    run()

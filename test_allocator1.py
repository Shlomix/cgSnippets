#!/usr/bin/env python3
"""
test_lazy_page_allocator_plain.py
=================================

Minimal, self-contained tests for lazy_page_allocator.py — *without* pytest.

Scenarios
---------
1. Prefill only                (no remotes should appear)
2. Prefill + decode, 1 remote  (exactly remote-0 promoted)
3. Prefill + decode, 2 remotes (remote-0 then remote-1 promoted)
4. Capacity check              (forcing promotion of 2nd remote)

Combinations: tp ∈ {1,2}  ×  pp ∈ {1,2}  ×  remotes ∈ {0,1,2}
"""

from lazy_page_allocator import LazyPageAllocator

# --------------------------------------------------------------------- #
def make_sequences(n_seq: int, tok_per_seq: int):
    """Return `n_seq` sequences, each with `tok_per_seq` dummy tokens."""
    return [list(range(tok_per_seq)) for _ in range(n_seq)]


def decode_steps(alloc: LazyPageAllocator, n_steps: int = 4):
    """Run `n_steps` decode iterations (1 token/seq/step)."""
    for _ in range(n_steps):
        alloc.step()


# --------------------------------------------------------------------- #
def run_tests():
    combos = [(tp, pp, rem)
              for tp in (1, 2)
              for pp in (1, 2)
              for rem in (0, 1, 2)]

    failures = []

    # ---------- Test 1: Prefill only ----------------------------------
    for tp, pp, rem in combos:
        seqs = make_sequences(4, 16)
        a = LazyPageAllocator(seqs, tp, pp, rem)
        if a.used_remote_worker_ids() != []:
            failures.append(f"PrefillOnly  tp={tp} pp={pp} rem={rem}")

    # ---------- Test 2: Prefill + decode, 1 remote --------------------
    for tp, pp in ((1,1), (2,1), (1,2), (2,2)):
        seqs = make_sequences(4, 16)
        a = LazyPageAllocator(seqs, tp, pp, remotes=1)
        decode_steps(a)                # 4 decode steps
        expected = [tp * pp]           # first remote’s ID
        if a.used_remote_worker_ids() != expected:
            failures.append(f"OneRemote  tp={tp} pp={pp}")

    # ---------- Test 3: Prefill + decode, 2 remotes -------------------
    for tp, pp in ((1,1), (2,1), (1,2), (2,2)):
        seqs = make_sequences(4, 16)
        a = LazyPageAllocator(seqs, tp, pp, remotes=2)
        decode_steps(a)
        base = tp * pp
        got = set(a.used_remote_worker_ids())
        want = {base, base + 1}
        if got != want:
            failures.append(f"TwoRemotes tp={tp} pp={pp} (got {got})")

    # ---------- Test 4: Capacity forces second remote -----------------
    for tp, pp in ((1,1), (2,1), (1,2), (2,2)):
        seqs = make_sequences(4, 16)
        a = LazyPageAllocator(seqs, tp, pp, remotes=2)
        if a.used_remote_worker_ids():        # after prefill
            failures.append(f"CapCheck-pre  tp={tp} pp={pp}")
            continue
        decode_steps(a)
        if len(a.used_remote_worker_ids()) != 2:
            failures.append(f"CapCheck-post tp={tp} pp={pp}")

    # ---------------- summary ----------------------------------------
    if failures:
        print("❌  Some tests failed:")
        for f in failures:
            print("   •", f)
    else:
        total = len(combos) + 3 * 4  # 18 distinct cases
        print(f"✅  All {total} cases passed!")


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    run_tests()

#!/usr/bin/env python3
from lazy_page_allocator import LazyPageAllocator

def make_sequences(n_seq, tok_per): return [list(range(tok_per)) for _ in range(n_seq)]
def decode_steps(a, n=4):           [a.step() for _ in range(n)]

def test_suite():
    combos = [(tp,pp,rem) for tp in (1,2) for pp in (1,2) for rem in (0,1,2)]
    fails  = []

    # 1. Prefill only
    for tp,pp,rem in combos:
        a = LazyPageAllocator(make_sequences(4,16), tp, pp, rem, warmed_up=True)
        if a.used_remote_worker_ids():
            fails.append(f"PrefillOnly tp={tp} pp={pp} rem={rem}")

    # 2. One remote after decode
    for tp,pp in ((1,1),(2,1),(1,2),(2,2)):
        a = LazyPageAllocator(make_sequences(4,16), tp, pp, remotes=1,
                              warmed_up=True, remote_max_tables=2)
        decode_steps(a)
        if a.used_remote_worker_ids() != [tp*pp]:
            fails.append(f"OneRemote tp={tp} pp={pp}")

    # 3. Two remotes after decode
    for tp,pp in ((1,1),(2,1),(1,2),(2,2)):
        a = LazyPageAllocator(make_sequences(4,16), tp, pp, remotes=2,
                              warmed_up=True, remote_max_tables=1)
        decode_steps(a)
        base = tp*pp
        if set(a.used_remote_worker_ids()) != {base, base+1}:
            fails.append(f"TwoRemotes tp={tp} pp={pp}")

    # 4. Capacity test (forces second remote)
    for tp,pp in ((1,1),(2,1),(1,2),(2,2)):
        a = LazyPageAllocator(make_sequences(4,16), tp, pp, remotes=2,
                              warmed_up=True, remote_max_tables=1)
        if a.used_remote_worker_ids(): fails.append(f"Cap-pre tp={tp} pp={pp}")
        decode_steps(a)
        if len(a.used_remote_worker_ids()) != 2:
            fails.append(f"Cap-post tp={tp} pp={pp}")

    # 5. NEW: locals should grow when remotes == 0
    a = LazyPageAllocator(make_sequences(4,16), tp=1, pp=1, remotes=0,
                          warmed_up=True)
    start_tables = len(a.shared_tables)
    decode_steps(a)
    if len(a.shared_tables) == start_tables:
        fails.append("LocalsGrow rem0")

    # summary
    if fails:
        print("❌  Failures:")
        for f in fails: print(" •", f)
    else:
        print(f"✅  All {len(combos)+3*4+1} cases passed!")  # 19 total

if __name__ == "__main__":
    test_suite()

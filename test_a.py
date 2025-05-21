#!/usr/bin/env python3
from lazy_page_allocator import LazyPageAllocator

def make_sequences(n_seq, ntok):
    return [list(range(ntok)) for _ in range(n_seq)]

def decode_steps(alloc, steps=4):
    for _ in range(steps):
        alloc.step()

def remote_cap(rem):
    return None if rem == 0 else (2 if rem == 1 else 1)

def test_suite():
    combos = [(tp, pp, rem) for tp in (1, 2) for pp in (1, 2) for rem in (0, 1, 2)]
    fails = []

    # 1) Prefill only
    for tp, pp, rem in combos:
        a = LazyPageAllocator(make_sequences(4, 16), tp, pp, rem,
                              remote_max_tables=remote_cap(rem))
        if a.used_remote_worker_ids():
            fails.append(f"PrefillOnly tp={tp} pp={pp} rem={rem}")

    # 2) One remote
    for tp, pp in ((1, 1), (2, 1), (1, 2), (2, 2)):
        base = tp * pp
        a = LazyPageAllocator(make_sequences(4, 16), tp, pp, remotes=1,
                              remote_max_tables=2)
        decode_steps(a)
        if a.used_remote_worker_ids() != [base]:
            fails.append(f"OneRemote tp={tp} pp={pp}")

    # 3) Two remotes
    for tp, pp in ((1, 1), (2, 1), (1, 2), (2, 2)):
        base = tp * pp
        a = LazyPageAllocator(make_sequences(4, 16), tp, pp, remotes=2,
                              remote_max_tables=1)
        decode_steps(a)
        if set(a.used_remote_worker_ids()) != {base, base + 1}:
            fails.append(f"TwoRemotes tp={tp} pp={pp}")

    # 4) Capacity forces second remote
    for tp, pp in ((1, 1), (2, 1), (1, 2), (2, 2)):
        a = LazyPageAllocator(make_sequences(4, 16), tp, pp, remotes=2,
                              remote_max_tables=1)
        if a.used_remote_worker_ids():
            fails.append(f"Cap-pre tp={tp} pp={pp}")
            continue
        decode_steps(a)
        if len(a.used_remote_worker_ids()) != 2:
            fails.append(f"Cap-post tp={tp} pp={pp}")

    # 5) Locals grow when remotes == 0
    a = LazyPageAllocator(make_sequences(4, 16), tp=1, pp=1, remotes=0)
    initial_tables = len(a.shared_tables)
    decode_steps(a)
    if len(a.shared_tables) == initial_tables:
        fails.append("LocalsGrow rem0")

    # 6) Remote-only scenario
    a = LazyPageAllocator("remotes_only", tp=1, pp=1, remotes=2,
                          remote_max_tables=2)
    # prefill should use only remote-0
    if a.used_remote_worker_ids() != [1]:
        fails.append("RemoteOnly prefill")
    decode_steps(a)
    if set(a.used_remote_worker_ids()) != {1, 2}:
        fails.append("RemoteOnly decode")

    # summary
    if fails:
        print("❌  Failures:")
        for f in fails:
            print(" •", f)
    else:
        total = len(combos) + 3 * 4 + 2   # 20 cases
        print(f"✅  All {total} cases passed!")

if __name__ == "__main__":
    test_suite()

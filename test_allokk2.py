#!/usr/bin/env python3
from lazy_page_allocator import LazyPageAllocator

def seqs(n, tok): return [list(range(tok)) for _ in range(n)]
def decode(a, k=4): [a.step() for _ in range(k)]
def rcap(r): return None if r==0 else (2 if r==1 else 1)

def run():
    combos = [(tp,pp,rem) for tp in (1,2) for pp in (1,2) for rem in (0,1,2)]
    bad = []

    # 1 Prefill only
    for tp,pp,rem in combos:
        a = LazyPageAllocator(seqs(4,16), tp, pp, rem,
                              remote_max_tables=rcap(rem))
        if a.used_remote_worker_ids(): bad.append(f"prefill tp{tp} pp{pp} r{rem}")

    # 2 One remote
    for tp,pp in ((1,1),(2,1),(1,2),(2,2)):
        base = tp*pp
        a = LazyPageAllocator(seqs(4,16), tp, pp, remotes=1,
                              remote_max_tables=2)
        decode(a)
        if a.used_remote_worker_ids() != [base]: bad.append(f"oneR tp{tp} pp{pp}")

    # 3 Two remotes
    for tp,pp in ((1,1),(2,1),(1,2),(2,2)):
        base = tp*pp
        a = LazyPageAllocator(seqs(4,16), tp, pp, remotes=2,
                              remote_max_tables=1)
        decode(a)
        if set(a.used_remote_worker_ids()) != {base,base+1}:
            bad.append(f"twoR tp{tp} pp{pp}")

    # 4 Capacity forces second remote
    for tp,pp in ((1,1),(2,1),(1,2),(2,2)):
        a = LazyPageAllocator(seqs(4,16), tp, pp, remotes=2,
                              remote_max_tables=1)
        if a.used_remote_worker_ids(): bad.append(f"cap-pre tp{tp} pp{pp}")
        decode(a)
        if len(a.used_remote_worker_ids()) != 2: bad.append(f"cap-post tp{tp} pp{pp}")

    # 5 Locals grow, remotes==0
    a = LazyPageAllocator(seqs(4,16), 1,1,0)
    t0 = len(a.shared_tables); decode(a); t1 = len(a.shared_tables)
    if t1 == t0: bad.append("locals-grow")

    # 6 Remote-only scenario
    a = LazyPageAllocator("remotes_only", 1,1, remotes=2, remote_max_tables=2)
    if a.used_remote_worker_ids()!=[1]: bad.append("RO prefill")
    decode(a);                                      # promote remote-1
    if set(a.used_remote_worker_ids())!={1,2}: bad.append("RO decode")

    # report
    if bad:
        print("❌ Fails:"); [print(" •", b) for b in bad]
    else:
        print("✅ All 20 cases passed!")

if __name__ == "__main__":
    run()

# create_test_jsons.py
import os, json, math, importlib.util, pathlib
from typing import List, Tuple

# --------------------------------------------------------------------------- #
# 1.  Capacity helper (page exclusivity respected)
# --------------------------------------------------------------------------- #
def plan_tables_for_run(
    *,
    seq_lengths: List[int],
    decode_steps: int,
    page_size: int,
    pages_per_table: int,
    remote_worker_ids: List[int],
    local_tables: int = 0,
    last_remote_grows: bool = True,
) -> List[Tuple[int, int]]:
    """Return minimal (worker_id, num_tables) list while keeping pages exclusive."""
    pages_per_seq = [
        math.ceil((prefill + decode_steps) / page_size) for prefill in seq_lengths
    ]
    total_pages  = sum(pages_per_seq)
    total_tables = math.ceil(total_pages / pages_per_table)

    order: List[Tuple[int, int]] = []
    if local_tables:
        order.append((-1, local_tables))
    rem_tables = max(0, total_tables - local_tables)

    if rem_tables and not remote_worker_ids:
        raise ValueError("Need remote tables but no remote workers supplied")

    if remote_worker_ids and rem_tables:
        if last_remote_grows:
            base = rem_tables // len(remote_worker_ids)
            extra = rem_tables % len(remote_worker_ids)
            for i, wid in enumerate(remote_worker_ids):
                order.append((wid, base + (extra if i == len(remote_worker_ids)-1 else 0)))
        else:
            per = math.ceil(rem_tables / len(remote_worker_ids))
            for wid in remote_worker_ids:
                order.append((wid, per))

    return order

# --------------------------------------------------------------------------- #
# 2.  Declare the five scenarios
# --------------------------------------------------------------------------- #
TESTS = {
    "local_only" : dict(
        input_ids       = [list(range(16))]*4,
        decode_tokens   = 4,
        page_size       = 16,
        pages_per_table = 2,
        table_order     = plan_tables_for_run(
            seq_lengths=[16]*4,
            decode_steps=4,
            page_size=16,
            pages_per_table=2,
            remote_worker_ids=[], local_tables=4),
    ),
    "local_and_1_remote" : dict(
        input_ids       = [list(range(16))]*4,
        decode_tokens   = 4,
        page_size       = 16,
        pages_per_table = 2,
        table_order     = plan_tables_for_run(
            seq_lengths=[16]*4,
            decode_steps=4,
            page_size=16,
            pages_per_table=2,
            remote_worker_ids=[0], local_tables=2),
    ),
    "local_and_2_remotes" : dict(
        input_ids       = [list(range(16))]*4,
        decode_tokens   = 4,
        page_size       = 16,
        pages_per_table = 2,
        table_order     = plan_tables_for_run(
            seq_lengths=[16]*4,
            decode_steps=4,
            page_size=16,
            pages_per_table=2,
            remote_worker_ids=[0,1], local_tables=1),
    ),
    "remotes_only_2" : dict(
        input_ids       = [list(range(16))]*4,
        decode_tokens   = 4,
        page_size       = 16,
        pages_per_table = 2,
        table_order     = plan_tables_for_run(
            seq_lengths=[16]*4,
            decode_steps=4,
            page_size=16,
            pages_per_table=2,
            remote_worker_ids=[0,1], local_tables=0),
    ),
    "sanity_local_bigtable" : dict(
        input_ids       = [list(range(16))]*64,
        decode_tokens   = 8,
        page_size       = 16,
        pages_per_table = 1536,
        table_order     = [(-1,1)],   # one gigantic table; helper not needed
    ),
}

# --------------------------------------------------------------------------- #
# 3.  Write JSON files
# --------------------------------------------------------------------------- #
os.makedirs("test_jsons", exist_ok=True)
for name, cfg in TESTS.items():
    with open(f"test_jsons/{name}.json", "w") as f:
        json.dump(cfg, f, indent=2)

# --------------------------------------------------------------------------- #
# 4.  Load allocator & run a quick simulation for each JSON
# --------------------------------------------------------------------------- #
def load_allocator():
    spec = importlib.util.spec_from_file_location(
        "page_allocator_streamed", pathlib.Path(__file__).with_name("page_allocator_streamed.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PageAllocatorStreamed

PageAllocator = load_allocator()

def run_once(cfg, name):
    table_log = []
    def alloc_table(wid:int)->int:
        tid = len(table_log)
        table_log.append((wid, tid))
        return tid

    PA = PageAllocator(
        page_size         = cfg["page_size"],
        pages_per_table   = cfg["pages_per_table"],
        table_allocation_order = cfg["table_order"],
        alloc_table       = alloc_table,
    )
    PA.warmup()
    PA.prefill(cfg["input_ids"])
    PA.step(len(cfg["input_ids"]))      # one decode iteration of 1-token/seq
    ok = True

    # check exclusivity
    seen=set()
    for seq in PA._seq_allocations:
        for p in seq:
            k=(p["table"],p["page"])
            if k in seen: ok=False
            seen.add(k)

    print(f"{name:<22} | tables={len(PA.table_id_to_worker):2} | pages={len(seen):3} | "
          f"workers={PA.workers_used()} | {'OK' if ok else '❌ SHARED'}")

if __name__ == "__main__":
    for name,cfg in TESTS.items():
        run_once(cfg, name)

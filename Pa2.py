import math
from typing import Dict, List

# ---------------------------------------------------------------------------
#  DummyBoostModel – stand‑in for your fem_boost_model
# ---------------------------------------------------------------------------
class DummyBoostModel:
    def __init__(self, start: int = 0):
        self._next = start

    def alloc_table(self) -> int:
        tid, self._next = self._next, self._next + 1
        return tid


# ---------------------------------------------------------------------------
#  LazyBatchAllocator  (prompt allocation + lazy growth, no worker_id in output)
# ---------------------------------------------------------------------------
class LazyBatchAllocator:
    """
    • Allocates all prompt pages at construction (fill worker‑0 first, etc.).
    • Creates tables lazily (fem_boost_model.alloc_table) exactly when needed.
    • step() adds +1 token to every sequence; pages are added on demand.
    • page_mappings returned to the caller have ONLY {"table_id","page_id"}.
    """

    def __init__(self,
                 input_ids: List[List[int]],
                 worker_table_counts: Dict[int, int],
                 fem_boost_model,
                 page_size: int,
                 table_capacity: int):

        # --- meta info ---------------------------------------------------
        self.B          = len(input_ids)
        if self.B == 0:
            raise ValueError("input_ids must contain at least one sequence")

        self.workers    = [w for w, n in sorted(worker_table_counts.items())
                           if n > 0]
        if not self.workers:
            raise ValueError("all workers have zero‑table quota")

        self.max_tables = worker_table_counts
        self.model      = fem_boost_model
        self.psize      = page_size
        self.cap        = table_capacity

        # --- dynamic per‑worker data ------------------------------------
        self.tables:   Dict[int, List[int]] = {w: [] for w in self.workers}
        self.used:     Dict[int, List[int]] = {w: [] for w in self.workers}
        self.next_pid: Dict[int, List[int]] = {w: [] for w in self.workers}

        self._table_owner: Dict[int, int]   = {}   # table_id → worker_id

        self.w_ptr = 0                               # round‑robin pointer

        # --- per‑sequence state -----------------------------------------
        self.tokens_seen   = [len(seq) for seq in input_ids]
        self.page_mappings = [[] for _ in range(self.B)]   # ⬅ user‑visible

        # allocate prompt pages
        for seq, tokens in enumerate(self.tokens_seen):
            for _ in range(math.ceil(tokens / self.psize)):
                self._allocate_one_page(seq)

    # ------------------------------------------------------------------ #
    # public: one decoding tick                                          #
    # ------------------------------------------------------------------ #
    def step(self) -> List[List[dict]]:
        for seq in range(self.B):
            prev = self.tokens_seen[seq]
            self.tokens_seen[seq] = prev + 1
            if prev % self.psize == 0:
                self._allocate_one_page(seq)
        return self.page_mappings

    # ------------------------------------------------------------------ #
    # internal: allocate exactly ONE page                                #
    # ------------------------------------------------------------------ #
    def _allocate_one_page(self, seq: int):
        nW = len(self.workers)

        for off in range(nW):
            wid = self.workers[(self.w_ptr + off) % nW]

            # 1️⃣  try existing, non‑full tables
            for idx, tid in enumerate(self.tables[wid]):
                if self.used[wid][idx] < self.cap:
                    self._place(seq, wid, idx, tid)
                    self.w_ptr = (self.w_ptr + off) % nW
                    return

            # 2️⃣  create new table if quota allows
            if len(self.tables[wid]) < self.max_tables[wid]:
                tid = self.model.alloc_table()
                self._table_owner[tid] = wid
                self.tables[wid].append(tid)
                self.used[wid].append(0)
                self.next_pid[wid].append(0)

                idx = len(self.tables[wid]) - 1
                self._place(seq, wid, idx, tid)
                self.w_ptr = (self.w_ptr + off) % nW
                return
            # else: worker full → try next

        raise RuntimeError("All workers are out of table capacity")

    # ------------------------------------------------------------------ #
    # helper: record placement (no worker_id exposed)                    #
    # ------------------------------------------------------------------ #
    def _place(self, seq: int, wid: int, idx: int, tid: int):
        pid = self.next_pid[wid][idx]
        self.page_mappings[seq].append({"table_id": tid, "page_id": pid})

        self.next_pid[wid][idx] += 1
        self.used[wid][idx]     += 1

 if __name__ == "__main__":
    batch   = [[1, 2, 3]] * 4           # 4 sequences (prompts = 3 tokens each)
    workers = {0: 2, 1: 0, 2: 1}
    PAGE_SZ = 4
    CAP     = 3

    alloc = LazyBatchAllocator(batch,
                               workers,
                               DummyBoostModel(),
                               page_size=PAGE_SZ,
                               table_capacity=CAP)

    print("Initial mapping:")
    for i, m in enumerate(alloc.page_mappings):
        print(f"  seq{i}: {m}")

    for step in range(5):
        try:
            mapping = alloc.step()
            print(f"\nstep {step+1}")
            for i, m in enumerate(mapping):
                print(f"  seq{i}: {m}")
        except RuntimeError as e:
            print(f"\nStopped: {e}")
            break 

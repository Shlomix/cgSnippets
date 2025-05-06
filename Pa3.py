import math
from typing import Dict, List

# ---------------------------------------------------------------------------
#  DummyBoostModel  – stand‑in for your real allocator
# ---------------------------------------------------------------------------
class DummyBoostModel:
    """Returns monotonically increasing table IDs; keeps last 'where' arg."""
    def __init__(self, start: int = 0):
        self._next = start
        self.calls = []                     # log of (worker_id, table_id)

    def alloc_table(self, where: int) -> int:
        tid, self._next = self._next, self._next + 1
        self.calls.append((where, tid))
        return tid


# ---------------------------------------------------------------------------
#  LazyBatchAllocator  (prompt allocation + lazy growth)
# ---------------------------------------------------------------------------
class LazyBatchAllocator:
    """
    • Prompt pages allocated during __init__ (tables created lazily).
    • step() adds one token to every sequence; allocates pages on demand.
    • Local worker is always ID ‑1.  Remote workers are ≥0.
    • page_mappings returned to the caller contain ONLY table_id & page_id.
    """

    LOCAL_WID = -1   # constant to mark the local worker

    def __init__(self,
                 input_ids: List[List[int]],
                 worker_table_counts: Dict[int, int],
                 fem_boost_model,
                 page_size: int,
                 table_capacity: int):
        """
        Parameters
        ----------
        input_ids            : list[list[int]]
        worker_table_counts  : {worker_id: max_tables}, must include LOCAL_WID
                               Workers with 0 tables are ignored.
        fem_boost_model      : object with alloc_table(where:int) -> int
        page_size            : tokens per page
        table_capacity       : pages a table can hold
        """
        if self.LOCAL_WID not in worker_table_counts:
            raise ValueError("worker_table_counts must include the local worker id (-1)")

        self.B = len(input_ids)
        if self.B == 0:
            raise ValueError("input_ids must contain at least one sequence")

        # keep workers with >0 quota, local first, remotes in ascending order
        self.workers = (
            [self.LOCAL_WID] if worker_table_counts[self.LOCAL_WID] > 0 else []
        ) + sorted([w for w, n in worker_table_counts.items()
                    if w >= 0 and n > 0])

        if not self.workers:
            raise ValueError("no worker owns any tables")

        self.max_tables = worker_table_counts
        self.model      = fem_boost_model
        self.psize      = page_size
        self.cap        = table_capacity

        # dynamic per‑worker data
        self.tables:   Dict[int, List[int]] = {w: [] for w in self.workers}
        self.used:     Dict[int, List[int]] = {w: [] for w in self.workers}
        self.next_pid: Dict[int, List[int]] = {w: [] for w in self.workers}

        self._table_owner: Dict[int, int] = {}  # table_id → worker_id

        self.w_ptr = 0  # index in self.workers (round‑robin pointer)

        # per‑sequence state
        self.tokens_seen   = [len(seq) for seq in input_ids]
        self.page_mappings = [[] for _ in range(self.B)]   # exposed to user

        # allocate prompt pages
        for seq, tok in enumerate(self.tokens_seen):
            for _ in range(math.ceil(tok / self.psize)):
                self._allocate_one_page(seq)

    # ------------------------------------------------------------------ #
    # public: decoding tick                                              #
    # ------------------------------------------------------------------ #
    def step(self) -> List[List[dict]]:
        for seq in range(self.B):
            prev = self.tokens_seen[seq]
            self.tokens_seen[seq] = prev + 1
            if prev % self.psize == 0:               # crossed boundary
                self._allocate_one_page(seq)
        return self.page_mappings

    # ------------------------------------------------------------------ #
    # internal: allocate ONE page                                        #
    # ------------------------------------------------------------------ #
    def _allocate_one_page(self, seq: int):
        nW = len(self.workers)

        for off in range(nW):
            wid = self.workers[(self.w_ptr + off) % nW]

            # 1️⃣  try existing tables
            for idx, tid in enumerate(self.tables[wid]):
                if self.used[wid][idx] < self.cap:
                    self._place(seq, wid, idx, tid)
                    self.w_ptr = (self.w_ptr + off) % nW
                    return

            # 2️⃣  create new table if quota allows
            if len(self.tables[wid]) < self.max_tables[wid]:
                tid = self._make_table(wid)
                idx = len(self.tables[wid]) - 1
                self._place(seq, wid, idx, tid)
                self.w_ptr = (self.w_ptr + off) % nW
                return
            # else: saturated → next worker

        raise RuntimeError("All workers are out of table capacity")

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    def _make_table(self, wid: int) -> int:
        """Create a fresh table for worker wid and return its id."""
        tid = self.model.alloc_table(wid)
        self._table_owner[tid] = wid
        self.tables[wid].append(tid)
        self.used[wid].append(0)
        self.next_pid[wid].append(0)
        return tid

    def _place(self, seq: int, wid: int, idx: int, tid: int):
        pid = self.next_pid[wid][idx]
        self.page_mappings[seq].append({"table_id": tid, "page_id": pid})
        self.next_pid[wid][idx] += 1
        self.used[wid][idx]     += 1

if __name__ == "__main__":
    input_ids = [[1, 2, 3]] * 4        # 4 sequences, each 3‑token prompt
    workers   = {-1: 2, 0: 0, 1: 1}    # local owns 2 tables, worker‑0 none
    PAGE_SZ   = 4
    CAP       = 3

    model = DummyBoostModel()
    alloc = LazyBatchAllocator(input_ids, workers, model,
                               page_size=PAGE_SZ, table_capacity=CAP)

    print("Initial mapping:")
    for i, m in enumerate(alloc.page_mappings):
        print(f"  seq{i}: {m}")

    # simulate decoding steps
    for step in range(6):
        try:
            mapping = alloc.step()
            print(f"\nstep {step+1}")
            for i, m in enumerate(mapping):
                print(f"  seq{i}: {m}")
        except RuntimeError as e:
            print(f"\nStopped: {e}")
            break

    print("\nalloc_table calls:", model.calls)

# ---------------------------------------------------------------------------
#  Stub for fem_boost_model (hands out table IDs)
# ---------------------------------------------------------------------------
class DummyBoostModel:
    def __init__(self, start: int = 0):
        self._next = start

    def alloc_table(self) -> int:
        tid, self._next = self._next, self._next + 1
        return tid


# ---------------------------------------------------------------------------
#  Lazy, worker‑first page allocator (one batch)
# ---------------------------------------------------------------------------
from typing import Dict, List


class SimpleLazyAllocator:
    """
    Allocate KV‑pages on demand, worker‑0 first, no infinite loops.

    Parameters
    ----------
    batch_size            : int               # number of sequences
    worker_table_counts   : Dict[int, int]    # {worker_id: max_tables}
    fem_boost_model       : object w/ .alloc_table()
    page_size             : int               # tokens per page
    table_capacity        : int               # pages a table can hold
    """

    # ---- construction ---------------------------------------------------
    def __init__(self,
                 batch_size: int,
                 worker_table_counts: Dict[int, int],
                 fem_boost_model,
                 page_size: int,
                 table_capacity: int):

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        # keep workers that actually own tables, sorted
        self.workers = [w for w, n in sorted(worker_table_counts.items())
                        if n > 0]
        if not self.workers:
            raise ValueError("all workers have zero‑table quota")

        self.max_tables  = worker_table_counts
        self.model       = fem_boost_model
        self.page_size   = page_size
        self.cap         = table_capacity

        # dynamic per‑worker data
        self.tables:      Dict[int, List[int]] = {w: [] for w in self.workers}
        self.used_pages:  Dict[int, List[int]] = {w: [] for w in self.workers}
        self.next_pageid: Dict[int, List[int]] = {w: [] for w in self.workers}

        # pointer to the worker we’ll try first next time
        self.worker_ptr = 0                     # index into self.workers

        # per‑sequence state
        self.tokens_seen   = [0] * batch_size
        self.page_mappings = [[] for _ in range(batch_size)]

    # ---- public: +1 token to every sequence -----------------------------
    def step(self):
        """
        Advance decoding by one token for every sequence.  Allocate a new
        page exactly when the added token is the first token in that page.

        Returns
        -------
        List[List[dict]]
            Full page mappings (one list per sequence).
        """
        for seq in range(len(self.tokens_seen)):
            prev = self.tokens_seen[seq]
            self.tokens_seen[seq] = prev + 1
            if prev % self.page_size == 0:          # new page needed
                self._allocate_one_page(seq)
        return self.page_mappings

    # ---- core allocation routine ---------------------------------------
    def _allocate_one_page(self, seq: int):
        """
        Find the first worker (starting from worker_ptr) with free space
        or quota for a new table.  Allocate ONE page there.  If nobody has
        room, raise RuntimeError.
        """
        num_workers = len(self.workers)

        for offset in range(num_workers):
            wid = self.workers[(self.worker_ptr + offset) % num_workers]

            # 1) try to place on an existing, NON‑FULL table
            for idx, tid in enumerate(self.tables[wid]):
                if self.used_pages[wid][idx] < self.cap:
                    self._place(seq, wid, idx, tid)
                    # remember this worker for the next call
                    self.worker_ptr = (self.worker_ptr + offset) % num_workers
                    return

            # 2) no free slot — can we create a new table for this worker?
            if len(self.tables[wid]) < self.max_tables[wid]:
                self._make_table(wid)              # adds an empty table
                idx = len(self.tables[wid]) - 1
                tid = self.tables[wid][idx]
                self._place(seq, wid, idx, tid)
                self.worker_ptr = (self.worker_ptr + offset) % num_workers
                return
            # else: this worker is saturated → try next worker

        # tried every worker once → no space left
        raise RuntimeError("All workers are out of table capacity")

    # ---- helpers --------------------------------------------------------
    def _make_table(self, wid: int):
        tid = self.model.alloc_table()
        self.tables[wid].append(tid)
        self.used_pages[wid].append(0)
        self.next_pageid[wid].append(0)

    def _place(self, seq: int, wid: int, idx: int, tid: int):
        """Record a single (worker, table, page) placement."""
        pid = self.next_pageid[wid][idx]
        self.page_mappings[seq].append(
            {"worker_id": wid, "table_id": tid, "page_id": pid}
        )
        # bump counters
        self.next_pageid[wid][idx] += 1
        self.used_pages[wid][idx]  += 1


if __name__ == "__main__":
    SEQS    = 4
    WORKERS = {0: 2, 1: 0, 2: 1}   # worker‑1 owns 0 tables
    PAGE_SZ = 4
    CAP     = 3

    alloc = SimpleLazyAllocator(SEQS, WORKERS,
                                DummyBoostModel(),
                                page_size=PAGE_SZ,
                                table_capacity=CAP)

    for step in range(12):  # will raise after capacity exhausted
        try:
            mapping = alloc.step()
            print(f"\nstep {step+1}")
            for s, m in enumerate(mapping):
                print(f"  seq{s}: {m}")
        except RuntimeError as e:
            print(f"\nAllocator halted at step {step+1}: {e}")
            break

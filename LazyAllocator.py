# ---------------------------------------------------------------------------
#  Minimal stub for fem_boost_model
# ---------------------------------------------------------------------------
class DummyBoostModel:
    """Returns monotonically increasing table IDs on demand."""
    def __init__(self, start: int = 0):
        self._next = start

    def alloc_table(self) -> int:              # called lazily
        tid, self._next = self._next, self._next + 1
        return tid


# ---------------------------------------------------------------------------
#  L a z y   P a g e d   A l l o c a t o r
# ---------------------------------------------------------------------------
from typing import Dict, List
from collections import defaultdict


class LazyPagedAllocator:
    """
    On‑demand, worker‑ordered KV‑page allocator for **one batch**.

    *   No tables are created in __init__; each appears only when its
        first page is placed.
    *   Current worker’s tables are filled to `table_capacity` before the
        pointer advances to the next worker.
    *   If every possible table is full, a RuntimeError is raised cleanly.

    Parameters
    ----------
    batch_size : int
        Number of sequences (seq_id is their index 0…batch_size‑1).
    worker_table_counts : Dict[int, int]
        {worker_id: max_tables_owned}.  Zero is allowed → worker skipped.
    fem_boost_model : object exposing .alloc_table() → int
    page_size : int
        Tokens per page.
    table_capacity : int
        Pages a table can hold.
    """

    # -------- construction -------------------------------------------------
    def __init__(self,
                 batch_size: int,
                 worker_table_counts: Dict[int, int],
                 fem_boost_model,
                 page_size: int,
                 table_capacity: int):

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.B                = batch_size
        self.workers          = [w for w in sorted(worker_table_counts)
                                 if worker_table_counts[w] > 0]
        if not self.workers:
            raise ValueError("No worker owns any tables")

        self.max_tables       = worker_table_counts
        self.model            = fem_boost_model
        self.page_size        = page_size
        self.table_capacity   = table_capacity

        # dynamic per‑worker structures (all start empty)
        self.tables_per_worker: Dict[int, List[int]] = defaultdict(list)
        self.used_pages:        Dict[int, List[int]] = defaultdict(list)
        self.next_page_id:      Dict[int, List[int]] = defaultdict(list)
        self.current_table_idx: Dict[int, int]       = defaultdict(int)  # per‑worker ptr

        # pointer to current worker (index in self.workers)
        self.worker_ptr = 0

        # per‑sequence state
        self.tokens_seen   = [0] * self.B
        self.page_mappings = [[] for _ in range(self.B)]

    # -------- public: advance decoding by ONE token ------------------------
    def step(self):
        """
        Give every sequence one more token.  A new page is allocated only
        when the fresh token is the first token of a page (i.e. when the
        previous total was 0, page_size, 2*page_size, …).

        Returns
        -------
        List[List[dict]]
            `page_mappings` – list[seq_id] → list of
            {"worker_id", "table_id", "page_id"}.
        """
        for seq in range(self.B):
            prev = self.tokens_seen[seq]
            self.tokens_seen[seq] = prev + 1
            if prev % self.page_size == 0:       # entering a new page
                self._allocate_one_page(seq)
        return self.page_mappings

    # -------- internals ----------------------------------------------------
    def _allocate_one_page(self, seq_id: int):
        """Find a free slot anywhere; allocate exactly ONE page there."""
        start_worker_idx = self.worker_ptr

        while True:
            wid = self.workers[self.worker_ptr]

            # ensure there's a current table for this worker
            if not self.tables_per_worker[wid]:
                self._make_new_table(wid)

            tidx = self.current_table_idx[wid]      # table index inside worker
            # if current table is full, move to next table (or next worker)
            if self.used_pages[wid][tidx] >= self.table_capacity:
                # try to create a new table if quota allows
                if len(self.tables_per_worker[wid]) < self.max_tables[wid]:
                    self._make_new_table(wid)
                    tidx = self.current_table_idx[wid]  # new table is last one
                else:
                    self._advance_worker()
                    if self.worker_ptr == start_worker_idx:
                        raise RuntimeError("All workers out of capacity")
                    continue  # loop again with new worker

            # we now have a table with free space
            tid   = self.tables_per_worker[wid][tidx]
            pid   = self.next_page_id[wid][tidx]

            self.page_mappings[seq_id].append(
                {"worker_id": wid,
                 "table_id":  tid,
                 "page_id":   pid}
            )

            # book‑keeping
            self.next_page_id[wid][tidx] += 1
            self.used_pages[wid][tidx]   += 1

            # if table just became full, update pointer for next time
            if self.used_pages[wid][tidx] >= self.table_capacity:
                self.current_table_idx[wid] += 1
                if self.current_table_idx[wid] >= len(self.tables_per_worker[wid]):
                    # will allocate or move worker on next call
                    self.current_table_idx[wid] = len(self.tables_per_worker[wid]) - 1

            return  # success

    def _make_new_table(self, wid: int):
        """Instantiate a fresh table for worker `wid`."""
        tid = self.model.alloc_table()
        self.tables_per_worker[wid].append(tid)
        self.used_pages[wid].append(0)
        self.next_page_id[wid].append(0)
        self.current_table_idx[wid] = len(self.tables_per_worker[wid]) - 1

    def _advance_worker(self):
        """Move the worker pointer to the next worker in round‑robin order."""
        self.worker_ptr = (self.worker_ptr + 1) % len(self.workers)

if __name__ == "__main__":
    SEQS    = 4
    WORKERS = {0: 2, 1: 0, 2: 1}   # worker‑1 owns zero tables
    PAGE_SZ = 4
    CAP     = 3

    alloc = LazyPagedAllocator(SEQS, WORKERS,
                               DummyBoostModel(),
                               page_size=PAGE_SZ,
                               table_capacity=CAP)

    for step in range(12):         # stop when RuntimeError occurs
        try:
            mapping = alloc.step()
            print(f"\nstep {step+1}")
            for s, m in enumerate(mapping):
                print(f"  seq{s}: {m}")
        except RuntimeError as e:
            print(f"\nAllocator halted at step {step+1}: {e}")
            break

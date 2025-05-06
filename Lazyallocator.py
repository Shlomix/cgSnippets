# ---------------------------------------------------------------------------
#  Dummy stand‑in for fem_boost_model
# ---------------------------------------------------------------------------
class DummyBoostModel:
    """Allocates monotonically increasing table IDs."""
    def __init__(self, start: int = 0):
        self._next = start

    def alloc_table(self) -> int:
        tid, self._next = self._next, self._next + 1
        return tid


# ---------------------------------------------------------------------------
#  Lazy (on‑demand) page allocator
# ---------------------------------------------------------------------------
import math
from typing import Dict, List


class LazyPagedAllocator:
    """
    vLLM‑style page allocator for ONE batch, creating tables *only when used*.

    * `batch_size`              – number of sequences (seq_id = 0 … batch‑1)
    * `worker_table_counts`     – {worker_id: max_tables_owned}
                                   → some workers may own 0 tables
    * `fem_boost_model`         – must expose .alloc_table()
    * `page_size`               – tokens per page
    * `table_capacity`          – pages a table can hold
    """

    def __init__(self,
                 batch_size: int,
                 worker_table_counts: Dict[int, int],
                 fem_boost_model,
                 page_size: int,
                 table_capacity: int):

        self.B                = batch_size
        self.page_size        = page_size
        self.table_capacity   = table_capacity
        self.model            = fem_boost_model

        # store workers in ascending order but allocate tables lazily
        self.workers          = sorted(worker_table_counts)
        self.max_tables       = worker_table_counts

        # per‑worker dynamic lists of table_ids (allocated on demand)
        self.tables_per_worker: Dict[int, List[int]] = {w: [] for w in self.workers}
        self.used_pages:        Dict[int, List[int]] = {w: [] for w in self.workers}
        self.next_page_id:      Dict[int, List[int]] = {w: [] for w in self.workers}

        # current worker / table indices
        self._worker_idx    = 0    # index into self.workers
        self._table_idx_in_w = 0   # which table inside current worker

        # per‑sequence information
        self.tokens_seen    = [0] * self.B
        self.page_mappings  = [[] for _ in range(self.B)]

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def step(self):
        """
        Add +1 token to every sequence. Allocate a new page **only** when
        the sequence crosses a page boundary. Returns the full page mapping.
        """
        for seq in range(self.B):
            prev = self.tokens_seen[seq]
            self.tokens_seen[seq] = prev + 1
            if prev % self.page_size == 0:        # entering a fresh page
                self._allocate_one_page(seq)
        return self.page_mappings

    # ------------------------------------------------------------------ #
    # internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _allocate_one_page(self, seq_id: int):
        """
        Place a single page according to fill‑current‑worker‑first policy,
        allocating new tables lazily as needed.
        """
        while True:
            wid = self.workers[self._worker_idx]

            # allocate first table for this worker if list is empty
            if not self.tables_per_worker[wid]:
                self._make_new_table(wid)

            # ensure _table_idx_in_w is within the current list length
            if self._table_idx_in_w >= len(self.tables_per_worker[wid]):
                # no more tables for this worker → move to next worker
                if self._advance_worker():
                    continue  # try again with next worker
                else:
                    raise RuntimeError("All tables in all workers are full")

            # indices now valid
            tid   = self.tables_per_worker[wid][self._table_idx_in_w]
            slot  = self._table_idx_in_w            # shorthand

            # if table is full try next table / worker
            if self.used_pages[wid][slot] >= self.table_capacity:
                if self._advance_table_or_worker(wid):
                    continue
                else:
                    raise RuntimeError("All tables in all workers are full")

            # --- we have a free slot on (wid, tid) ----------------------
            page_id = self.next_page_id[wid][slot]

            self.page_mappings[seq_id].append(
                {"worker_id": wid,
                 "table_id":  tid,
                 "page_id":   page_id}
            )

            # bookkeeping
            self.next_page_id[wid][slot] += 1
            self.used_pages[wid][slot]   += 1

            # if table became full, advance pointer for next call
            if self.used_pages[wid][slot] >= self.table_capacity:
                self._advance_table_or_worker(wid)

            return  # page placed → done

    # ------------------------------------------------------------------ #
    # pointer‑management helpers                                         #
    # ------------------------------------------------------------------ #
    def _make_new_table(self, wid: int):
        """Allocate a fresh table for worker `wid` (lazy)."""
        if len(self.tables_per_worker[wid]) >= self.max_tables[wid]:
            return False  # cannot make more
        tid = self.model.alloc_table()
        self.tables_per_worker[wid].append(tid)
        self.used_pages[wid].append(0)
        self.next_page_id[wid].append(0)
        return True

    def _advance_table_or_worker(self, wid: int):
        """Try next table in same worker; if none, advance to next worker."""
        self._table_idx_in_w += 1
        # allocate a new table lazily if capacity allows
        if self._table_idx_in_w >= len(self.tables_per_worker[wid]):
            if self._make_new_table(wid):          # created new table
                return True
            else:
                return self._advance_worker()
        return True

    def _advance_worker(self):
        """Move to next worker that owns ≥1 table (existing or alloc‑able)."""
        start = self._worker_idx
        while True:
            self._worker_idx = (self._worker_idx + 1) % len(self.workers)
            self._table_idx_in_w = 0
            wid = self.workers[self._worker_idx]
            # if worker has capacity to host at least 1 table, ok
            if self.max_tables[wid] > 0:
                if not self.tables_per_worker[wid]:
                    self._make_new_table(wid)
                return True
            if self._worker_idx == start:
                return False  # made a full loop – no space anywhere

if __name__ == "__main__":
    SEQS     = 4
    WORKERS  = {0: 2, 1: 0, 2: 1}   # worker‑1 owns 0 tables
    PAGE_SZ  = 4
    CAP      = 3                    # small to see rotation

    alloc = LazyPagedAllocator(SEQS,
                               WORKERS,
                               DummyBoostModel(),
                               page_size=PAGE_SZ,
                               table_capacity=CAP)

    for step in range(10):
        mapping = alloc.step()
        print(f"\nstep {step+1}")
        for s, m in enumerate(mapping):
            print(f"  seq{s}: {m}")

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
#  LazyPagedAllocator  –  tables created on demand, no infinite loop
# ---------------------------------------------------------------------------
from collections import defaultdict
from typing import Dict, List


class LazyPagedAllocator:
    """
    Single‑batch, on‑demand KV‑page allocator.

    • Tables are instantiated (model.alloc_table) only when a page is first
      placed on them.
    • Current worker’s tables are filled before moving to the next worker.
    • If every possible table in every worker is full, a RuntimeError is
      raised instead of looping indefinitely.
    """

    def __init__(self,
                 batch_size: int,
                 worker_table_counts: Dict[int, int],
                 fem_boost_model,
                 page_size: int,
                 table_capacity: int):
        """
        Parameters
        ----------
        batch_size : int
            Number of sequences in the batch (seq_id is 0 … batch_size‑1)
        worker_table_counts : {worker_id: max_tables_owned}
            Workers may own 0 tables; workers appear in numerical order.
        fem_boost_model : object with .alloc_table() → int
        page_size : int
            Tokens per page
        table_capacity : int
            Pages a table can hold
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.B                = batch_size
        self.page_size        = page_size
        self.table_capacity   = table_capacity
        self.model            = fem_boost_model

        # workers sorted numerically; each has a max‑table quota
        self.workers          = sorted(worker_table_counts)
        self.max_tables       = worker_table_counts

        # dynamic per‑worker data
        self.tables_per_worker: Dict[int, List[int]] = {w: [] for w in self.workers}
        self.used_pages:        Dict[int, List[int]] = {w: [] for w in self.workers}
        self.next_page_id:      Dict[int, List[int]] = {w: [] for w in self.workers}

        # pointers
        self._worker_idx     = 0      # index in self.workers
        self._table_idx_in_w = 0      # table offset inside current worker

        # per‑sequence state
        self.tokens_seen     = [0] * self.B
        self.page_mappings   = [[] for _ in range(self.B)]

    # ------------------------------------------------------------------ #
    # public: advance decoding by +1 token for EVERY sequence            #
    # ------------------------------------------------------------------ #
    def step(self):
        """
        Give every sequence one more token.  Allocate a new page only when
        the added token is the first token of that page (0‑based counts:
        0, page_size, 2*page_size …).  Returns the full page_mappings.
        """
        for seq in range(self.B):
            prev = self.tokens_seen[seq]
            self.tokens_seen[seq] = prev + 1
            if prev % self.page_size == 0:
                self._allocate_one_page(seq)
        return self.page_mappings

    # ------------------------------------------------------------------ #
    # internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _allocate_one_page(self, seq_id: int):
        tried_all = False
        while True:
            wid = self.workers[self._worker_idx]

            # Ensure the current worker has at least one table slot
            if not self.tables_per_worker[wid]:
                if not self._maybe_make_table(wid):
                    tried_all = self._advance_worker(tried_all)
                    continue

            # current table indices valid?
            if self._table_idx_in_w >= len(self.tables_per_worker[wid]):
                tried_all = self._advance_worker(tried_all)
                continue

            slot = self._table_idx_in_w
            if self.used_pages[wid][slot] >= self.table_capacity:
                # table full → try next table (or worker)
                tried_all = self._advance_table_or_worker(wid, tried_all)
                continue

            # --- place the page -----------------------------------------
            tid     = self.tables_per_worker[wid][slot]
            page_id = self.next_page_id[wid][slot]

            self.page_mappings[seq_id].append(
                {"worker_id": wid,
                 "table_id":  tid,
                 "page_id":   page_id}
            )

            # book‑keeping
            self.next_page_id[wid][slot] += 1
            self.used_pages[wid][slot]   += 1

            # if table became full, move pointer so next allocation looks elsewhere
            if self.used_pages[wid][slot] >= self.table_capacity:
                self._advance_table_or_worker(wid)

            return  # success

    # ------------ pointer / table management --------------------------- #
    def _maybe_make_table(self, wid: int) -> bool:
        """Create a new table for worker wid if quota allows."""
        if len(self.tables_per_worker[wid]) >= self.max_tables[wid]:
            return False
        tid = self.model.alloc_table()
        self.tables_per_worker[wid].append(tid)
        self.used_pages[wid].append(0)
        self.next_page_id[wid].append(0)
        return True

    def _advance_table_or_worker(self, wid: int, tried_all=False) -> bool:
        """Move to next table in same worker; if none, advance worker."""
        self._table_idx_in_w += 1
        if self._table_idx_in_w < len(self.tables_per_worker[wid]):
            return False  # stayed in same worker
        # need a new table or move worker
        if self._maybe_make_table(wid):
            return False
        return self._advance_worker(tried_all)

    def _advance_worker(self, tried_all=False) -> bool:
        """
        Move to next worker.  If we've already looped all workers and found
        no space, raise RuntimeError (only once).
        """
        start = self._worker_idx
        while True:
            self._worker_idx = (self._worker_idx + 1) % len(self.workers)
            self._table_idx_in_w = 0
            wid = self.workers[self._worker_idx]

            # skip workers with 0‑table quota
            if self.max_tables[wid] == 0:
                if self._worker_idx == start:
                    break        # full loop, no space
                continue

            # ensure at least one table exists or can be created
            if self.tables_per_worker[wid] or self._maybe_make_table(wid):
                return False     # moved to a worker with space

            if self._worker_idx == start:
                break            # checked everyone

        if tried_all:
            raise RuntimeError("All workers are out of table capacity")
        return self._advance_worker(True)  # one recursive retry


# ---------------------------------------------------------------------------
#  Tiny demo (will throw RuntimeError when fully exhausted)                  #
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SEQS    = 4
    WORKERS = {0: 2, 1: 0, 2: 1}   # worker‑1 owns 0 tables
    PAGE_SZ = 4
    CAP     = 3

    alloc = LazyPagedAllocator(SEQS, WORKERS,
                               DummyBoostModel(),
                               page_size=PAGE_SZ,
                               table_capacity=CAP)

    for step in range(12):
        try:
            mapping = alloc.step()
            print(f"\nstep {step+1}")
            for s, m in enumerate(mapping):
                print(f"  seq{s}: {m}")
        except RuntimeError as e:
            print(f"\nStopped at step {step+1}: {e}")
            break

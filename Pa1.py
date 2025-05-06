import math
from typing import Dict, List


# ---------------------------------------------------------------------------
#  DummyBoostModel  – stand‑in for your fem_boost_model
# ---------------------------------------------------------------------------
class DummyBoostModel:
    """Hands out monotonically increasing table IDs when asked."""
    def __init__(self, start: int = 0):
        self._next = start

    def alloc_table(self) -> int:
        tid, self._next = self._next, self._next + 1
        return tid


# ---------------------------------------------------------------------------
#  LazyBatchAllocator
# ---------------------------------------------------------------------------
class LazyBatchAllocator:
    """
    vLLM‑style page allocator (single batch).

    * **Prompt pages** are allocated during __init__.
    * Tables are created lazily as soon as the first page lands on them.
    * During decoding, call `step()` to add one token per sequence; pages
      are allocated lazily when new boundaries are crossed.
    * Worker‑0 is filled completely before worker‑1, etc.  Workers with a
      zero‑table quota are ignored automatically.
    """

    # ---------- construction -------------------------------------------
    def __init__(self,
                 input_ids: List[List[int]],
                 worker_table_counts: Dict[int, int],
                 fem_boost_model,
                 page_size: int,
                 table_capacity: int):

        # -------- basic checks -----------------------------------------
        self.B = len(input_ids)
        if self.B == 0:
            raise ValueError("input_ids must contain at least one sequence")

        # keep workers that actually own tables, sorted numerically
        self.workers = [wid for wid, n in sorted(worker_table_counts.items())
                        if n > 0]
        if not self.workers:
            raise ValueError("all workers have zero‑table quota")

        self.max_tables = worker_table_counts
        self.model      = fem_boost_model
        self.page_size  = page_size
        self.cap        = table_capacity

        # dynamic per‑worker structures
        self.tables:   Dict[int, List[int]] = {w: [] for w in self.workers}
        self.used:     Dict[int, List[int]] = {w: [] for w in self.workers}
        self.next_pid: Dict[int, List[int]] = {w: [] for w in self.workers}

        # pointer to the worker we’ll start with next allocation
        self.w_ptr = 0  # index in self.workers

        # per‑sequence state
        self.tokens_seen   = [len(seq) for seq in input_ids]
        self.page_mappings = [[] for _ in range(self.B)]

        # -------- prompt allocation ------------------------------------
        for seq_id, tokens in enumerate(self.tokens_seen):
            pages_needed = math.ceil(tokens / self.page_size)
            for _ in range(pages_needed):
                self._allocate_one_page(seq_id)

    # ---------- public: +1 token to every sequence ----------------------
    def step(self) -> List[List[dict]]:
        """
        Advance decoding by one token for every sequence.  A new page is
        allocated only when the fresh token is the first token of that page.

        Returns
        -------
        list[list[dict]]  –  full page mapping (seq_id → list of dicts).
        """
        for seq in range(self.B):
            prev = self.tokens_seen[seq]
            self.tokens_seen[seq] = prev + 1
            if prev % self.page_size == 0:              # crossed boundary
                self._allocate_one_page(seq)
        return self.page_mappings

    # ---------- internal: allocate ONE page -----------------------------
    def _allocate_one_page(self, seq: int):
        """
        Find a worker/table with free space (creating a table lazily if
        quota allows) and place exactly one page.  Raises RuntimeError if
        no worker can accept more pages.
        """
        n_workers = len(self.workers)

        for offset in range(n_workers):
            wid = self.workers[(self.w_ptr + offset) % n_workers]

            # 1) try existing non‑full tables in this worker
            for idx, tid in enumerate(self.tables[wid]):
                if self.used[wid][idx] < self.cap:
                    self._place(seq, wid, idx, tid)
                    self.w_ptr = (self.w_ptr + offset) % n_workers
                    return

            # 2) no free slot – create a new table if quota permits
            if len(self.tables[wid]) < self.max_tables[wid]:
                self._make_table(wid)
                idx = len(self.tables[wid]) - 1
                tid = self.tables[wid][idx]
                self._place(seq, wid, idx, tid)
                self.w_ptr = (self.w_ptr + offset) % n_workers
                return
            # else: worker saturated → try next worker

        raise RuntimeError("All workers are out of table capacity")

    # ---------- helpers -------------------------------------------------
    def _make_table(self, wid: int):
        tid = self.model.alloc_table()
        self.tables[wid].append(tid)
        self.used[wid].append(0)
        self.next_pid[wid].append(0)

    def _place(self, seq: int, wid: int, idx: int, tid: int):
        pid = self.next_pid[wid][idx]
        self.page_mappings[seq].append(
            {"worker_id": wid, "table_id": tid, "page_id": pid}
        )
        self.next_pid[wid][idx] += 1
        self.used[wid][idx]     += 1

if __name__ == "__main__":
    batch   = [[1, 2, 3]] * 4           # 4 sequences, 3‑token prompts
    workers = {0: 2, 1: 0, 2: 1}        # worker‑1 owns 0 tables
    PAGE_SZ = 4
    CAP     = 3

    alloc = LazyBatchAllocator(batch,
                               workers,
                               DummyBoostModel(),
                               page_size=PAGE_SZ,
                               table_capacity=CAP)

    print("Initial mapping (prompts allocated):")
    for s, m in enumerate(alloc.page_mappings):
        print(f"  seq{s}: {m}")

    # simulate decoding
    for step in range(6):
        try:
            mapping = alloc.step()
            print(f"\nstep {step+1}")
            for s, m in enumerate(mapping):
                print(f"  seq{s}: {m}")
        except RuntimeError as e:
            print(f"\nAllocator stopped at step {step+1}: {e}")
            break

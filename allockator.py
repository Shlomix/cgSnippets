# allocator_with_factory.py
import math


class PageAllocator:
    """
    Minimal page allocator
    ----------------------
    * page_size, pages_per_table in tokens
    * remote_worker_ids – order matters (promotion happens only when the
      previous worker’s table is full)
    * table_factory.alloc_table(worker_id) -> table_id
        worker_id == -1  → local table
        worker_id >= 0   → remote table on that worker
    * API: warmup()  prefill(seq_lens)  decode()
    """

    # ─────────────────────────────── init ────────────────────────────────
    def __init__(
        self,
        *,
        page_size: int,
        pages_per_table: int,
        remote_worker_ids: list[int],
        decode_tokens: int,
        table_factory,
        locals_present: bool = True,
    ):
        self.P = page_size
        self.T = pages_per_table
        self.RW = remote_worker_ids                # ordered!
        self.decode_tokens = decode_tokens
        self.factory = table_factory
        self.locals_present = locals_present

        # runtime
        self.tables: dict[int, dict] = {}          # tid → {"remote":bool, "pages_used":int, "worker":int}
        self.curr_remote_idx = 0                   # index in RW we’re filling now
        self.curr_remote_tid: int | None = None    # current remote table id
        self.seq_pages: list[list[dict]] = []      # per-sequence page refs
        self.n: int | None = None

        # planning
        self.locals = None
        self.remotes_target = None                 # total remote tables we expect
        self._planned = False
        self._warmup_page = None

    # ───────────────── private helpers ─────────────────
    def _plan(self, seq_lens: list[int]):
        S0 = sum(seq_lens)
        n = len(seq_lens)
        total_pages = math.ceil((S0 + n * self.decode_tokens) / self.P)

        # locals -----------------------------------------------------------
        if not self.locals_present:
            self.locals = 0
        else:
            self.locals = math.ceil(S0 / (self.T * self.P))

        # ensure at least one remote page will spill if remotes exist
        rem_pages = max(0, total_pages - self.locals * self.T)
        if self.locals_present and self.RW and rem_pages == 0:
            rem_pages = 1

        # number of remote tables we MIGHT allocate (upper bound)
        if not self.RW:
            self.remotes_target = 0
        else:
            self.remotes_target = max(
                math.ceil(rem_pages / self.T), len(self.RW)
            )

    def _new_table(self, remote: bool, worker_id: int):
        tid = self.factory.alloc_table(worker_id)
        self.tables[tid] = {"remote": remote, "pages_used": 0, "worker": worker_id}
        if remote:
            self.curr_remote_tid = tid
        return tid

    def _ensure_local_tables(self):
        while sum(not t["remote"] for t in self.tables.values()) < self.locals:
            self._new_table(False, -1)

    def _advance_remote_worker(self):
        """Move curr_remote_idx to the next worker (cyclic)."""
        self.curr_remote_idx = (self.curr_remote_idx + 1) % len(self.RW)
        worker_id = self.RW[self.curr_remote_idx]
        return self._new_table(True, worker_id)

    def _ensure_remote_table(self):
        """Make sure there is a remote table with free space."""
        if not self.RW:
            raise RuntimeError("no remote workers configured")

        if self.curr_remote_tid is None:           # first spill
            wid = self.RW[self.curr_remote_idx]
            self._new_table(True, wid)

        tb = self.tables[self.curr_remote_tid]
        if tb["pages_used"] >= self.T:             # table full → promote
            self._advance_remote_worker()

    def _alloc_page(self):
        # try locals first
        for tid, tb in self.tables.items():
            if not tb["remote"] and tb["pages_used"] < self.T:
                pid = tb["pages_used"]; tb["pages_used"] += 1
                return {"table_id": tid, "page_id": pid}

        # need remote page
        self._ensure_remote_table()
        tid = self.curr_remote_tid
        tb = self.tables[tid]
        pid = tb["pages_used"]; tb["pages_used"] += 1
        return {"table_id": tid, "page_id": pid}

    # ───────────────────── public API ─────────────────────
    def warmup(self):
        """Touch exactly one page and return its ref (pages_used stays 0)."""
        if self._warmup_page is not None:
            return self._warmup_page

        # create first table lazily in the tier that will serve first
        if not self.tables:
            if self.locals_present:
                self._new_table(False, -1)
            elif self.RW:
                self._new_table(True, self.RW[0])
            else:
                self._new_table(False, -1)

        tid = next(iter(self.tables))
        self._warmup_page = {"table_id": tid, "page_id": 0}
        return self._warmup_page

    def prefill(self, seq_lens: list[int]):
        if not self._planned:
            self._plan(seq_lens)
            self._ensure_local_tables()
            self.n = len(seq_lens)
            self.seq_pages = [[] for _ in range(self.n)]
            self._planned = True

        for i, tok in enumerate(seq_lens):
            for _ in range(math.ceil(tok / self.P)):
                self.seq_pages[i].append(self._alloc_page())
        return self.state()

    def decode(self):
        if not self._planned:
            raise RuntimeError("prefill first")
        pages_each = math.ceil(self.decode_tokens / self.P)
        for i in range(self.n):
            for _ in range(pages_each):
                self.seq_pages[i].append(self._alloc_page())
        return self.state()

    def state(self):
        return [list(p) for p in self.seq_pages]


# ───────────────────── Dummy factory + tests ─────────────────────
class DummyFactory:
    """Allocates incremental IDs and records (tid, worker_id)."""
    def __init__(self):
        self.next_id = 0
        self.calls = []

    def alloc_table(self, worker_id: int) -> int:
        tid = self.next_id
        self.next_id += 1
        self.calls.append((tid, worker_id))
        return tid


def _run_case(num_remotes: int, use_warmup: bool):
    factory = DummyFactory()

    alloc = PageAllocator(
        page_size=16,
        pages_per_table=2,
        remote_worker_ids=list(range(num_remotes)),   # [0,1,2,…] order matters
        decode_tokens=1,
        table_factory=factory,
        locals_present=True,
    )

    warm = alloc.warmup() if use_warmup else None
    pre = alloc.prefill([16, 16, 16, 16])            # 4×16 tokens

    # locals must equal 2 (64 tokens → 4 pages → 2 tables *local*)
    assert alloc.locals == 2, f"locals={alloc.locals}"

    # remote tables promotion: table IDs should appear grouped by worker
    if num_remotes:
        remote_calls = [c for c in factory.calls if c[1] >= 0]
        # each worker should appear once before any repeats
        seen_workers = []
        for _, wid in remote_calls:
            if wid not in seen_workers:
                seen_workers.append(wid)
            else:                                    # repeat
                assert len(seen_workers) == len(alloc.RW), \
                    "promoted new worker too early"

    # warm-up page reused
    if use_warmup:
        assert pre[0][0] == warm, "warm-up page not overwritten"

    # worker-id correctness
    for tid, wid in factory.calls:
        if alloc.tables[tid]["remote"]:
            assert wid >= 0
        else:
            assert wid == -1


def _run_tests():
    for r in (0, 1, 2):
        for warm in (False, True):
            _run_case(r, warm)
    print("All tests passed ✅")


if __name__ == "__main__":
    _run_tests()

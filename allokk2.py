"""
lazy_page_allocator.py
----------------------

Highlights
----------
• Explicit `warmup()` — must be called *before* `prefill()` if you need it.
• Optional `auto_prefill` flag (default True) keeps old behaviour for tests.
• Locals mirror tables; spill to remotes incrementally.
• Locals grow only when remotes == 0.
• Remote-only mode: sequences="remotes_only".
• model.allocate_table(worker_id): -1 for locals, remote-id otherwise.
"""

from __future__ import annotations
from math import ceil
from typing import Dict, List, Optional

# ─────────────── stub model (swap in your real one) ──────────────────────
class Model:
    def __init__(self) -> None:
        self._ctr = 0
    def allocate_table(self, worker_id: int) -> str:
        return f"tbl{worker_id}_{self._ctr:=self._ctr+1}"

# ─────────────────────────── low-level objects ───────────────────────────
class Page:
    def __init__(self, seq: int, table: "Table", sz: int):
        self.seq_id, self.table, self.page_size = seq, table, sz
        self.tokens_used = 0
    def remaining(self):           return self.page_size - self.tokens_used
    def add_tokens(self, n):       self.tokens_used += n
    @property
    def page_id(self):             return f"{self.table.table_id}_{self.table.pages.index(self)}"

class Table:
    def __init__(self, tid: str, owner: int, idx: int, ppt: int):
        self.table_id, self.owner_id, self.idx = tid, owner, idx
        self.pages_per_table = ppt
        self.pages: List[Page] = []
    def has_space(self):           return len(self.pages) < self.pages_per_table
    def add_page(self, seq: int, sz: int):
        if not self.has_space(): raise RuntimeError("table full")
        p = Page(seq, self, sz); self.pages.append(p); return p

class Worker:
    def __init__(self, wid: int, max_tables: Optional[int]):
        self.worker_id, self.max_tables = wid, max_tables
        self.tables: List[Table] = []
    def has_table_cap(self):       return self.max_tables is None or len(self.tables) < self.max_tables
    def add_table(self, t: Table): self.tables.append(t)
    def find_with_space(self):     return next((t for t in self.tables if t.has_space()), None)
    def is_empty(self):            return all(len(t.pages) == 0 for t in self.tables)

# ─────────────────────────── allocator core ──────────────────────────────
class LazyPageAllocator:
    def __init__(
        self,
        sequences,                 # List[List[int]] or "remotes_only"
        tp: int,
        pp: int,
        remotes: int,
        *,
        page_size=16,
        pages_per_table=2,
        auto_prefill=True,         # ← default keeps old behaviour
        remote_max_tables=None,
        local_ids=None,
        remote_ids=None,
        model=None,
    ):
        self.page_size, self.pages_per_table = page_size, pages_per_table
        self.remote_only = sequences == "remotes_only"
        if self.remote_only:
            sequences = [list(range(page_size)) for _ in range(4)]
        self.sequences, self.num_sequences = sequences, len(sequences)
        self.model = model or Model()

        # IDs -------------------------------------------------------------
        self.local_cnt = tp * pp
        self.local_ids = local_ids or list(range(self.local_cnt))
        self.remote_ids = remote_ids or list(
            range(self.local_cnt, self.local_cnt + remotes)
        )
        if len(self.local_ids) != self.local_cnt or len(self.remote_ids) != remotes:
            raise ValueError("ID list mismatch")

        # locals ---------------------------------------------------------
        pre_tables = ceil(self.num_sequences / pages_per_table)
        cap = 0 if self.remote_only else (None if remotes == 0 else pre_tables)
        self.locals = [Worker(wid, cap) for wid in self.local_ids]
        self.shared_tables: List[Table] = []
        for idx in range(pre_tables if cap != 0 else 0):
            tid = self.model.allocate_table(-1)
            tbl = Table(tid, -1, idx, pages_per_table)
            self.shared_tables.append(tbl)
            for w in self.locals: w.add_table(tbl)

        # remotes --------------------------------------------------------
        def_cap = 2 if remotes == 1 else 1
        self.remote_max_tables = def_cap if remote_max_tables is None else remote_max_tables
        self.remotes_: Dict[int, Worker] = {}
        self._next_remote = 0

        # bookkeeping ----------------------------------------------------
        self._seq_pages: Dict[int, List[Page]] = {}
        self._prefill_done = False
        self._warmup_done = False

        if auto_prefill:
            self.prefill()          # old behaviour retained for tests

    # ─────────────── public lifecycle helpers ──────────────────────────
    def warmup(self, short_seq: List[int], repeats: int = 1):
        """Write `short_seq` onto the SAME page `repeats` times."""
        if self._warmup_done: raise RuntimeError("warmup() already called")
        if self._prefill_done: raise RuntimeError("warmup after prefill")
        if len(short_seq) > self.page_size:
            raise ValueError("short_seq longer than page_size")

        pages = [self._last_page(s) or self._allocate_new_page(s)
                 for s in range(self.num_sequences)]

        for _ in range(repeats):
            for p in pages: p.add_tokens(len(short_seq)); p.tokens_used = 0

        self._warmup_done = True
        return [[{"table_id": p.table.table_id, "page_id": p.page_id}] for p in pages]

    def prefill(self):
        """Allocate real sequences (run once)."""
        if self._prefill_done: return
        for s, seq in enumerate(self.sequences): self._add_tokens(s, len(seq))
        self._prefill_done = True

    def step(self):
        if not self._prefill_done: self.prefill()   # lazily ensure
        for s in range(self.num_sequences): self._add_tokens(s, 1)
        return [[{"table_id": p.table.table_id, "page_id": p.page_id}
                 for p in self._seq_pages.get(s, [])]
                for s in range(self.num_sequences)]

    def used_remote_worker_ids(self):
        return [wid for wid in self.remote_ids
                if wid in self.remotes_ and not self.remotes_[wid].is_empty()]

    # ────────────────── internal allocation helpers ───────────────────
    def _last_page(self, s): return self._seq_pages.get(s, [])[-1] if s in self._seq_pages else None

    def _add_tokens(self, s, n):
        left = n
        while left:
            pg = self._last_page(s)
            if pg and pg.remaining():
                take = min(left, pg.remaining()); pg.add_tokens(take); left -= take; continue
            pg = self._alloc_new_page(s)
            take = min(left, pg.remaining()); pg.add_tokens(take); left -= take

    def _alloc_new_page(self, s):
        # locals
        for t in self.shared_tables:
            if t.has_space(): return self._record(t.add_page(s, self.page_size))
        # locals grow if no remotes
        if not self.remote_ids:
            tid = self.model.allocate_table(-1)
            idx = len(self.shared_tables)
            tbl = Table(tid, -1, idx, self.pages_per_table)
            self.shared_tables.append(tbl)
            for w in self.locals: w.add_table(tbl)
            return self._record(tbl.add_page(s, self.page_size))
        # remotes
        while self._next_remote < len(self.remote_ids):
            wid = self.remote_ids[self._next_remote]
            w   = self.remotes_.setdefault(wid, Worker(wid, self.remote_max_tables))
            tbl = w.find_with_space()
            if tbl is None and w.has_table_cap():
                tid = self.model.allocate_table(wid)
                tbl = Table(tid, wid, len(w.tables), self.pages_per_table)
                w.add_table(tbl)
            if tbl: return self._record(tbl.add_page(s, self.page_size))
            self._next_remote += 1
        raise RuntimeError("Out of capacity")

    def _record(self, p: Page): self._seq_pages.setdefault(p.seq_id, []).append(p); return p

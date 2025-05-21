"""
lazy_page_allocator.py  –  final version
----------------------------------------

• Mirrored locals, lazy spill-over to remotes
• Locals may grow during decode if remotes == 0
• Optional `warmed_up=True` pre-allocates one page/sequence
• Tunables: page_size, pages_per_table, remote_max_tables, explicit IDs
• NEW:  model.allocate_table(<worker_id>)  is called with
        -1            for local/mirrored tables
         remote_id    for remote-owned tables
"""

from __future__ import annotations
from math import ceil
from typing import Dict, List, Optional

# ───────────── stub “Model” that now expects a worker_id argument ─────────
class Model:
    def __init__(self) -> None:
        self._counter = 0
    def allocate_table(self, worker_id: int) -> str:
        """
        worker_id == -1 → local mirrored table
        worker_id >= 0  → remote-owned table with that ID
        """
        tid = f"tbl{worker_id}_{self._counter}"
        self._counter += 1
        return tid

# ───────────────────────────── primitives ────────────────────────────────
class Page:
    def __init__(self, seq: int, table: "Table", page_size: int):
        self.seq_id, self.table, self.page_size = seq, table, page_size
        self.tokens_used = 0
    def remaining(self):  return self.page_size - self.tokens_used
    def add_tokens(self, n):  self.tokens_used += n
    @property
    def page_id(self):  return f"{self.table.table_id}_{self.table.pages.index(self)}"

class Table:
    def __init__(self, tid: str, owner: int, idx: int, ppt: int):
        self.table_id, self.owner_id, self.idx = tid, owner, idx
        self.pages_per_table = ppt
        self.pages: List[Page] = []
    def has_space(self):  return len(self.pages) < self.pages_per_table
    def add_page(self, seq: int, sz: int) -> Page:
        if not self.has_space():  raise RuntimeError("table full")
        p = Page(seq, self, sz); self.pages.append(p); return p

class Worker:
    def __init__(self, wid: int, max_tables: Optional[int]):
        self.worker_id, self.max_tables = wid, max_tables
        self.tables: List[Table] = []
    def has_table_capacity(self):  return self.max_tables is None or len(self.tables) < self.max_tables
    def add_table(self, t: Table):  self.tables.append(t)
    def find_table_with_space(self):  return next((t for t in self.tables if t.has_space()), None)
    def is_empty(self):  return all(len(t.pages) == 0 for t in self.tables)

# ─────────────────────────── allocator ───────────────────────────────────
class LazyPageAllocator:
    def __init__(
        self,
        sequences: List[List[int]],
        tp: int,
        pp: int,
        remotes: int,
        *,
        page_size: int = 16,
        pages_per_table: int = 2,
        warmed_up: bool = False,
        remote_max_tables: Optional[int] = None,
        local_ids: Optional[List[int]] = None,
        remote_ids: Optional[List[int]] = None,
        model: Optional[Model] = None,
    ):
        # tunables / data -------------------------------------------------
        self.page_size, self.pages_per_table = page_size, pages_per_table
        self.sequences, self.num_sequences = sequences, len(sequences)
        self.model = model or Model()

        # worker IDs ------------------------------------------------------
        self.local_cnt = tp * pp
        self.local_ids = local_ids or list(range(self.local_cnt))
        if len(self.local_ids) != self.local_cnt:
            raise ValueError("len(local_ids) must equal tp*pp")

        self.remote_ids = remote_ids or list(
            range(self.local_cnt, self.local_cnt + remotes)
        )
        if len(self.remote_ids) != remotes:
            raise ValueError("len(remote_ids) must equal `remotes`")

        # locals: mirrored tables ----------------------------------------
        tables_needed = ceil(self.num_sequences / self.pages_per_table)
        cap = None if remotes == 0 else tables_needed
        self.locals = [Worker(wid, max_tables=cap) for wid in self.local_ids]

        self.shared_tables: List[Table] = []
        for idx in range(tables_needed):
            tid = self.model.allocate_table(-1)          # local table id
            tbl = Table(tid, owner=-1, idx=idx, ppt=self.pages_per_table)
            self.shared_tables.append(tbl)
            for w in self.locals: w.add_table(tbl)

        # remotes ---------------------------------------------------------
        default_remote_cap = 2 if remotes == 1 else 1
        self.remote_max_tables = (
            remote_max_tables if remote_max_tables is not None else default_remote_cap
        )
        self.remotes_: Dict[int, Worker] = {}
        self._next_remote_idx = 0

        # bookkeeping -----------------------------------------------------
        self._seq_pages: Dict[int, List[Page]] = {}

        # optional warm-up skeleton --------------------------------------
        if warmed_up:
            for s in range(self.num_sequences):
                self._record(self._allocate_new_page(s))  # empty page

        # prefill ---------------------------------------------------------
        self._prefill()

    # ───────────────────────── public API ─────────────────────────────
    def step(self):
        for s in range(self.num_sequences): self._add_tokens(s, 1)
        return [[{"table_id": p.table.table_id, "page_id": p.page_id}
                 for p in self._seq_pages.get(s, [])]
                for s in range(self.num_sequences)]

    def used_remote_worker_ids(self):
        return [wid for wid in self.remote_ids
                if wid in self.remotes_ and not self.remotes_[wid].is_empty()]

    # ───────────────────── internal helpers ───────────────────────────
    def _prefill(self):  [self._add_tokens(i, len(seq)) for i, seq in enumerate(self.sequences)]
    def _last_page(self, s): return self._seq_pages.get(s, [])[-1] if s in self._seq_pages else None

    def _add_tokens(self, s, n):
        left = n
        while left:
            pg = self._last_page(s)
            if pg and pg.remaining():
                take = min(left, pg.remaining()); pg.add_tokens(take); left -= take; continue
            pg = self._allocate_new_page(s)
            take = min(left, pg.remaining()); pg.add_tokens(take); left -= take

    def _allocate_new_page(self, s) -> Page:
        # 1) locals first
        for t in self.shared_tables:
            if t.has_space():
                return self._record(t.add_page(s, self.page_size))

        # 1b) if no remotes, locals may grow
        if not self.remote_ids:
            tid = self.model.allocate_table(-1)
            idx = len(self.shared_tables)
            new_tbl = Table(tid, owner=-1, idx=idx, ppt=self.pages_per_table)
            self.shared_tables.append(new_tbl)
            for w in self.locals: w.add_table(new_tbl)
            return self._record(new_tbl.add_page(s, self.page_size))

        # 2) spill to remotes lazily
        while self._next_remote_idx < len(self.remote_ids):
            wid = self.remote_ids[self._next_remote_idx]
            w   = self.remotes_.setdefault(wid, Worker(wid, self.remote_max_tables))
            tbl = w.find_table_with_space()
            if tbl is None and w.has_table_capacity():
                tid = self.model.allocate_table(wid)
                tbl = Table(tid, owner=wid, idx=len(w.tables), ppt=self.pages_per_table)
                w.add_table(tbl)
            if tbl: return self._record(tbl.add_page(s, self.page_size))
            self._next_remote_idx += 1
        raise RuntimeError("Out of capacity across locals + remotes")

    def _record(self, p: Page): self._seq_pages.setdefault(p.seq_id, []).append(p); return p

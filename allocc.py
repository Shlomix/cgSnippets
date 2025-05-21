"""
lazy_page_allocator.py
----------------------

Core abstractions
----------------
Page  -> belongs to exactly one sequence
Table -> fixed-size list of pages, lives on one worker
Worker-> local (mirrored tables) or remote (private tables)

Key behaviours
--------------
• Locals mirror the same Table objects; capacity = prefill need, unless
  remotes==0 (then unlimited).  They never grow during decode *unless* there
  are no remotes.
• Remotes are promoted lazily, one at a time.  Each remote’s capacity is
  remote_max_tables × pages_per_table pages.
• Passing  sequences="remotes_only"  disables locals entirely.
• Warm-up is an *explicit* call; it always re-uses the first allocated page
  per sequence and never allocates new pages.

Public API
----------
LazyPageAllocator(
        sequences, tp, pp, remotes,
        page_size=16, pages_per_table=2,
        remote_max_tables=None,
        local_ids=None, remote_ids=None,
        model=None)

warmup(short_seq, repeats=1)  -> per-sequence state
step()                        -> per-sequence state
used_remote_worker_ids()      -> list[int]
"""

from __future__ import annotations
from math import ceil
from typing import Dict, List, Optional

# ──────────────────────── stub model for table IDs ───────────────────────
class Model:
    def __init__(self) -> None:
        self._counter = 0

    def allocate_table(self, worker_id: int) -> str:
        """
        worker_id == -1  → local mirrored table
        worker_id >= 0   → table owned by that remote worker
        """
        tid = f"tbl{worker_id}_{self._counter}"
        self._counter += 1
        return tid

# ───────────────────────────── primitives ────────────────────────────────
class Page:
    def __init__(self, seq: int, table: "Table", page_size: int):
        self.seq_id, self.table, self.page_size = seq, table, page_size
        self.tokens_used = 0

    def remaining(self) -> int:
        return self.page_size - self.tokens_used

    def add_tokens(self, n: int) -> None:
        if n > self.remaining():
            raise ValueError("page overflow")
        self.tokens_used += n

    @property
    def page_id(self) -> str:
        return f"{self.table.table_id}_{self.table.pages.index(self)}"


class Table:
    def __init__(self, tid: str, owner: int, idx: int, ppt: int):
        self.table_id, self.owner_id, self.idx = tid, owner, idx
        self.pages_per_table = ppt
        self.pages: List[Page] = []

    def has_space(self) -> bool:
        return len(self.pages) < self.pages_per_table

    def add_page(self, seq: int, sz: int) -> Page:
        if not self.has_space():
            raise RuntimeError("table full")
        p = Page(seq, self, sz)
        self.pages.append(p)
        return p


class Worker:
    def __init__(self, wid: int, max_tables: Optional[int]):
        self.worker_id, self.max_tables = wid, max_tables
        self.tables: List[Table] = []

    def has_table_capacity(self) -> bool:
        return self.max_tables is None or len(self.tables) < self.max_tables

    def add_table(self, t: Table) -> None:
        self.tables.append(t)

    def find_table_with_space(self) -> Optional[Table]:
        return next((t for t in self.tables if t.has_space()), None)

    def is_empty(self) -> bool:
        return all(len(t.pages) == 0 for t in self.tables)


# ─────────────────────────── allocator core ──────────────────────────────
class LazyPageAllocator:
    def __init__(
        self,
        sequences,                 # List[List[int]]  or  "remotes_only"
        tp: int,
        pp: int,
        remotes: int,
        *,
        page_size: int = 16,
        pages_per_table: int = 2,
        remote_max_tables: Optional[int] = None,
        local_ids: Optional[List[int]] = None,
        remote_ids: Optional[List[int]] = None,
        model: Optional[Model] = None,
    ):
        # sentinel: remote-only mode (no local tables)
        self.remote_only = sequences == "remotes_only"
        if self.remote_only:
            sequences = [list(range(page_size)) for _ in range(4)]  # dummy

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

        # local workers ---------------------------------------------------
        prefill_tables = ceil(self.num_sequences / self.pages_per_table)
        cap = 0 if self.remote_only else (None if remotes == 0 else prefill_tables)
        self.locals = [Worker(wid, max_tables=cap) for wid in self.local_ids]

        self.shared_tables: List[Table] = []
        for idx in range(prefill_tables if cap != 0 else 0):
            tid = self.model.allocate_table(-1)
            tbl = Table(tid, owner=-1, idx=idx, ppt=self.pages_per_table)
            self.shared_tables.append(tbl)
            for w in self.locals:
                w.add_table(tbl)

        # remotes ---------------------------------------------------------
        default_remote_cap = 2 if remotes == 1 else 1
        self.remote_max_tables = (
            default_remote_cap if remote_max_tables is None else remote_max_tables
        )
        self.remotes_: Dict[int, Worker] = {}
        self._next_remote_idx = 0

        # bookkeeping -----------------------------------------------------
        self._seq_pages: Dict[int, List[Page]] = {}
        self._warmup_done = False

        # real prefill ----------------------------------------------------
        self._prefill()

    # ───────────────────────── public API ──────────────────────────────
    def warmup(self, short_seq: List[int], repeats: int = 1):
        """
        Write `short_seq` onto the SAME first page for each sequence,
        `repeats` times.  Requires len(short_seq) ≤ page_size.
        No new pages are allocated during warm-up.
        Returns per-sequence state.
        """
        if self._warmup_done:
            raise RuntimeError("warmup() already called")

        if len(short_seq) > self.page_size:
            raise ValueError(
                f"short_seq longer than page_size "
                f"({len(short_seq)} > {self.page_size})"
            )

        pages: List[Page] = []
        for seq_id in range(self.num_sequences):
            pg = self._last_page(seq_id)
            if pg is None:
                pg = self._allocate_new_page(seq_id)
            pages.append(pg)

        for _ in range(repeats):
            for pg in pages:
                pg.add_tokens(len(short_seq))
                pg.tokens_used = 0  # reset for next repeat

        state = [
            [{"table_id": pg.table.table_id, "page_id": pg.page_id}]
            for pg in pages
        ]
        self._warmup_done = True
        return state

    # ------------------------------------------------------------------
    def step(self):
        for s in range(self.num_sequences):
            self._add_tokens(s, 1)
        return [
            [{"table_id": p.table.table_id, "page_id": p.page_id}
             for p in self._seq_pages.get(s, [])]
            for s in range(self.num_sequences)
        ]

    # ------------------------------------------------------------------
    def used_remote_worker_ids(self) -> List[int]:
        return [
            wid for wid in self.remote_ids
            if wid in self.remotes_ and not self.remotes_[wid].is_empty()
        ]

    # ──────────────────── internal helpers ─────────────────────────────
    def _prefill(self):
        for s, seq in enumerate(self.sequences):
            self._add_tokens(s, len(seq))

    def _last_page(self, s):  # type: ignore[override]
        return self._seq_pages.get(s, [])[-1] if s in self._seq_pages else None

    def _add_tokens(self, s, n):
        left = n
        while left:
            pg = self._last_page(s)
            if pg and pg.remaining():
                take = min(left, pg.remaining())
                pg.add_tokens(take)
                left -= take
                continue
            pg = self._allocate_new_page(s)
            take = min(left, pg.remaining())
            pg.add_tokens(take)
            left -= take

    def _allocate_new_page(self, s) -> Page:
        # 1) locals (if any)
        for t in self.shared_tables:
            if t.has_space():
                return self._record(t.add_page(s, self.page_size))

        # 1b) locals may grow only when there are NO remotes
        if not self.remote_ids:
            tid = self.model.allocate_table(-1)
            idx = len(self.shared_tables)
            new_tbl = Table(tid, owner=-1, idx=idx, ppt=self.pages_per_table)
            self.shared_tables.append(new_tbl)
            for w in self.locals: w.add_table(new_tbl)
            return self._record(new_tbl.add_page(s, self.page_size))

        # 2) spill to remotes
        while self._next_remote_idx < len(self.remote_ids):
            wid = self.remote_ids[self._next_remote_idx]
            w = self.remotes_.setdefault(wid, Worker(wid, self.remote_max_tables))
            tbl = w.find_table_with_space()
            if tbl is None and w.has_table_capacity():
                tid = self.model.allocate_table(wid)
                tbl = Table(tid, owner=wid, idx=len(w.tables), ppt=self.pages_per_table)
                w.add_table(tbl)
            if tbl:
                return self._record(tbl.add_page(s, self.page_size))
            self._next_remote_idx += 1

        raise RuntimeError("Out of capacity")

    def _record(self, p: Page) -> Page:
        self._seq_pages.setdefault(p.seq_id, []).append(p)
        return p

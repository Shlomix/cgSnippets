"""
lazy_page_allocator.py
----------------------

A flexible “lazy page allocator” that:

* Mirrors tables across *local* workers (tp × pp)
* Lazily promotes *remote* workers one-by-one when locals are full
* Supports an optional warm-up phase whose pages are overwritten
* Works with arbitrary page sizes, pages-per-table, worker-ID lists
* Exposes two public methods:

      warmup(short_sequence)  →  per-sequence state   (only once)
      step()                  →  per-sequence state   (decode 1 token/seq)

  …where “state” is a list (one entry per sequence) of
      { "table_id": <str>, "page_id": <str> }  dictionaries.
"""

from __future__ import annotations

from math import ceil
from typing import Dict, List, Optional


# ─────────────────────────── external “model” ────────────────────────────
class Model:
    """Tiny stub that hands out globally unique table IDs."""
    def __init__(self) -> None:
        self._counter = 0

    def allocate_table(self) -> str:
        tid = f"tbl_{self._counter}"
        self._counter += 1
        return tid


# ───────────────────────────── primitives ────────────────────────────────
class Page:
    def __init__(self, seq_id: int, table: "Table", page_size: int):
        self.seq_id = seq_id
        self.table = table
        self.page_size = page_size
        self.tokens_used = 0

    # helpers -------------------------------------------------------------
    def remaining(self) -> int:
        return self.page_size - self.tokens_used

    def add_tokens(self, n: int) -> None:
        if n > self.remaining():
            raise ValueError("page overflow")
        self.tokens_used += n

    @property
    def page_id(self) -> str:  # unique within a table
        idx = self.table.pages.index(self)
        return f"{self.table.table_id}_{idx}"


class Table:
    def __init__(
        self,
        table_id: str,
        owner_id: int,
        idx: int,
        pages_per_table: int,
    ):
        self.table_id = table_id
        self.owner_id = owner_id          # worker id (locals share one id)
        self.idx = idx                    # index on that worker
        self.pages_per_table = pages_per_table
        self.pages: List[Page] = []

    # helpers -------------------------------------------------------------
    def has_space(self) -> bool:
        return len(self.pages) < self.pages_per_table

    def add_page(self, seq_id: int, page_size: int) -> Page:
        if not self.has_space():
            raise RuntimeError("table full")
        p = Page(seq_id, self, page_size)
        self.pages.append(p)
        return p


class Worker:
    """Either a local (mirrored tables) or a remote (private tables) worker."""

    def __init__(self, wid: int, max_tables: Optional[int]):
        self.worker_id = wid
        self.max_tables = max_tables  # None ⇒ unlimited
        self.tables: List[Table] = []

    # helpers -------------------------------------------------------------
    def has_table_capacity(self) -> bool:
        return self.max_tables is None or len(self.tables) < self.max_tables

    def add_table(self, table: Table) -> None:
        if not self.has_table_capacity():
            raise RuntimeError("worker out of table slots")
        self.tables.append(table)

    def find_table_with_space(self) -> Optional[Table]:
        for t in self.tables:
            if t.has_space():
                return t
        return None

    def is_empty(self) -> bool:
        return all(len(t.pages) == 0 for t in self.tables)


# ─────────────────────────── lazy allocator ──────────────────────────────
class LazyPageAllocator:
    """
    Parameters
    ----------
    sequences : List[List[int]]
        Real sequences (prefill).  One page (page_size tokens) is enough.
    tp, pp, remotes : int
        Tensor-, pipeline-parallelism factors & number of remote workers.
    page_size : int, default 16
    pages_per_table : int, default 2
    warmup_sequence : Optional[List[int]]
        Short sequence (< page_size) duplicated for every sequence *once*.
    local_ids : Optional[List[int]]
        Explicit IDs for locals (len == tp*pp).  Defaults: 0…tp*pp-1.
    remote_ids : Optional[List[int]]
        Explicit IDs for remotes (len == remotes).  Defaults: consecutive ints
        right after locals, used in that order for promotion.
    model : Optional[Model]
        Object exposing allocate_table().  Stub provided if omitted.
    """

    # ───────────────────────── constructor ────────────────────────────
    def __init__(
        self,
        sequences: List[List[int]],
        tp: int,
        pp: int,
        remotes: int,
        *,
        page_size: int = 16,
        pages_per_table: int = 2,
        warmup_sequence: Optional[List[int]] = None,
        local_ids: Optional[List[int]] = None,
        remote_ids: Optional[List[int]] = None,
        model: Optional[Model] = None,
    ):
        # tunables & data -------------------------------------------------
        self.page_size = page_size
        self.pages_per_table = pages_per_table
        self.sequences = sequences
        self.num_sequences = len(sequences)

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

        # model handle ----------------------------------------------------
        self.model = model or Model()

        # locals: fixed capacity (mirrored tables) ------------------------
        tables_needed = ceil(self.num_sequences / self.pages_per_table)
        self.locals: List[Worker] = [
            Worker(wid, max_tables=tables_needed) for wid in self.local_ids
        ]
        self.shared_tables: List[Table] = []
        for idx in range(tables_needed):
            t = Table(
                self.model.allocate_table(),
                owner_id=self.local_ids[0],
                idx=idx,
                pages_per_table=self.pages_per_table,
            )
            self.shared_tables.append(t)
            for w in self.locals:
                w.add_table(t)          # same object reference (mirrored)

        # remotes (lazy) --------------------------------------------------
        self.remote_max_tables = 2 if remotes == 1 else 1
        self.remotes_: Dict[int, Worker] = {}   # wid → Worker
        self._next_remote_idx = 0               # promotion pointer

        # bookkeeping -----------------------------------------------------
        self._seq_pages: Dict[int, List[Page]] = {}
        self._warmed = False                    # prevent double warm-up
        self.warmup_state: Optional[
            List[List[Dict[str, str]]]
        ] = None  # exposed if warm-up is run

        # optional warm-up (returns state) --------------------------------
        if warmup_sequence is not None:
            self.warmup_state = self.warmup(warmup_sequence)

        # prefill (real sequences) ----------------------------------------
        self._prefill()

    # =====================================================================
    #  PUBLIC API
    # =====================================================================
    def warmup(self, short_seq: List[int]) -> List[List[Dict[str, str]]]:
        """
        Perform warm-up allocation (< page_size tokens per sequence),
        then *overwrite* (zero) token counters so real prefill can reuse pages.

        Returns
        -------
        state : per-sequence list of {"table_id", "page_id"} dicts
        """
        if self._warmed:
            raise RuntimeError("warmup() may be called only once")

        # allocate warm-up tokens
        for seq_id in range(self.num_sequences):
            self._add_tokens(seq_id, len(short_seq))

        # capture state BEFORE zeroing
        state = self._build_state()

        # overwrite tokens (keep pages)
        for pages in self._seq_pages.values():
            for p in pages:
                p.tokens_used = 0

        self._warmed = True
        return state

    # ---------------------------------------------------------------------
    def step(self) -> List[List[Dict[str, str]]]:
        """
        Decode **one token per sequence**, allocate as needed,
        and return the updated per-sequence state.
        """
        for seq_id in range(self.num_sequences):
            self._add_tokens(seq_id, 1)
        return self._build_state()

    # ---------------------------------------------------------------------
    def used_remote_worker_ids(self) -> List[int]:
        """IDs of remote workers that currently hold any pages (order kept)."""
        return [
            wid
            for wid in self.remote_ids
            if wid in self.remotes_ and not self.remotes_[wid].is_empty()
        ]

    # =====================================================================
    #  internal helpers
    # =====================================================================
    def _build_state(self) -> List[List[Dict[str, str]]]:
        state: List[List[Dict[str, str]]] = []
        for seq_id in range(self.num_sequences):
            pages = self._seq_pages.get(seq_id, [])
            state.append(
                [
                    {"table_id": p.table.table_id, "page_id": p.page_id}
                    for p in pages
                ]
            )
        return state

    # ------------------------------------------------------------------ #
    def _prefill(self) -> None:
        for seq_id, seq in enumerate(self.sequences):
            self._add_tokens(seq_id, len(seq))

    # ------------------------------------------------------------------ #
    def _last_page(self, seq_id: int) -> Optional[Page]:
        pages = self._seq_pages.get(seq_id)
        return pages[-1] if pages else None

    # ------------------------------------------------------------------ #
    def _add_tokens(self, seq_id: int, n_tokens: int) -> None:
        left = n_tokens
        while left:
            page = self._last_page(seq_id)
            if page and page.remaining():
                take = min(left, page.remaining())
                page.add_tokens(take)
                left -= take
                continue

            page = self._allocate_new_page(seq_id)
            take = min(left, page.remaining())
            page.add_tokens(take)
            left -= take

    # ------------------------------------------------------------------ #
    def _allocate_new_page(self, seq_id: int) -> Page:
        # 1) try any local mirrored table
        for tbl in self.shared_tables:
            if tbl.has_space():
                return self._record(tbl.add_page(seq_id, self.page_size))

        # no more local capacity → spill to remotes lazily
        while self._next_remote_idx < len(self.remote_ids):
            wid = self.remote_ids[self._next_remote_idx]
            remote = self.remotes_.setdefault(
                wid, Worker(wid, self.remote_max_tables)
            )

            tbl = remote.find_table_with_space()
            if tbl is None and remote.has_table_capacity():
                tbl = Table(
                    self.model.allocate_table(),
                    owner_id=wid,
                    idx=len(remote.tables),
                    pages_per_table=self.pages_per_table,
                )
                remote.add_table(tbl)

            if tbl is not None:
                return self._record(tbl.add_page(seq_id, self.page_size))

            # this remote is full → promote next one
            self._next_remote_idx += 1

        raise RuntimeError("🔥  Out of capacity across locals + remotes!")

    # ------------------------------------------------------------------ #
    def _record(self, page: Page) -> Page:
        self._seq_pages.setdefault(page.seq_id, []).append(page)
        return page

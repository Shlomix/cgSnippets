# page_allocator_streamed.py
from typing import List, Tuple, Dict, Callable, Optional


class PageBudgetExceeded(Exception):
    """Raised when no more (table, page) slots remain."""
    pass


class PageAllocatorStreamed:
    """
    A streamed page allocator.

    •   Tables are supplied up-front via `table_allocation_order`:
        List[ (worker_id, num_tables) ].

    •   A *page* (table, page_idx) is never shared between sequences.
        A *table* can host pages from many sequences.

    •   warmup() allocates exactly one page (table=0, page=0) and returns it.
        It is reused as the first page of sequence 0 during prefill().

    •   prefill() (re)initialises all internal state and returns, for every
        input sequence, the **full list** of pages now owned by that sequence.

    •   step() appends one token per sequence, allocating new pages only when
        the current private page for that sequence is full.  It returns the
        **complete running history** (same structure as prefill()).
    """

    # ---------- construction -------------------------------------------------
    def __init__(
        self,
        page_size: int,
        pages_per_table: int,
        table_allocation_order: List[Tuple[int, int]],   # (worker_id, num_tables)
        alloc_table: Callable[[int], int],               # returns table_id
    ):
        self.page_size = page_size
        self.pages_per_table = pages_per_table
        self.alloc_table = alloc_table
        self.table_allocation_order = table_allocation_order

        # iterator state over the global table/page stream
        self._worker_idx = 0
        self._table_idx = 0
        self._page_idx = 0

        # realised tables  {(worker_id, table_idx) → table_id}
        self._tables: Dict[Tuple[int, int], int] = {}
        self.table_id_to_worker: Dict[int, int] = {}

        # sequence-level state (filled by warmup/prefill)
        self._seq_allocations: List[List[Dict]] = []         # pages per sequence
        self._seq_current_page: List[Optional[Dict]] = []    # last page for seq
        self._seq_tokens_used: List[int] = []                # tokens in that page

        self._warmup_page: Optional[Dict] = None

    # ---------- internal helpers --------------------------------------------
    def _next_unused_page(self) -> Dict:
        """Return the next free (table, page) slot, creating tables lazily."""
        while self._worker_idx < len(self.table_allocation_order):
            wid, tables_here = self.table_allocation_order[self._worker_idx]

            # exhausted the tables on this worker → move to next worker
            if self._table_idx >= tables_here:
                self._worker_idx += 1
                self._table_idx = 0
                self._page_idx = 0
                continue

            # exhausted pages in this table → move to next table on same worker
            if self._page_idx >= self.pages_per_table:
                self._table_idx += 1
                self._page_idx = 0
                continue

            # realise table if first time seen
            key = (wid, self._table_idx)
            if key not in self._tables:
                tbl_id = self.alloc_table(wid)
                self._tables[key] = tbl_id
                self.table_id_to_worker[tbl_id] = wid

            tbl_id = self._tables[key]
            pg_id = self._page_idx
            self._page_idx += 1
            return {"table": tbl_id, "page": pg_id}

        # ran through every table / page slot
        raise PageBudgetExceeded("Exhausted configured tables/pages")

    def _allocate_tokens_for_seq(self, tokens: int, idx: int) -> List[Dict]:
        """Allocate *tokens* for sequence *idx*, reusing its private page."""
        pages: List[Dict] = []
        while tokens > 0:
            cur_page = self._seq_current_page[idx]
            used     = self._seq_tokens_used[idx]

            # need a fresh page?
            if cur_page is None or used == self.page_size:
                cur_page = self._next_unused_page()
                self._seq_current_page[idx] = cur_page
                self._seq_tokens_used[idx]  = 0
                used = 0
                pages.append(cur_page)               # add **once** per page

            fill = min(tokens, self.page_size - used)
            self._seq_tokens_used[idx] += fill
            tokens -= fill

        return pages

    # ---------- public API ---------------------------------------------------
    def warmup(self) -> List[List[Dict]]:
        """Allocate the first page (table 0 / page 0)."""
        if self._warmup_page is None:
            self._warmup_page = self._next_unused_page()
            self._seq_allocations = [[self._warmup_page]]
            self._seq_current_page = [self._warmup_page]
            self._seq_tokens_used  = [0]             # no tokens yet
        return self._seq_allocations

    def prefill(self, sequences: List[List[int]]) -> List[List[Dict]]:
        """
        Allocate pages for each input sequence, **overwriting** any warm-up
        token counts but *reusing* the warm-up page for sequence 0.
        """
        self._seq_allocations.clear()
        self._seq_current_page.clear()
        self._seq_tokens_used.clear()

        for i, seq in enumerate(sequences):
            if i == 0 and self._warmup_page is not None:
                # reuse warm-up page as first page for seq-0
                self._seq_allocations.append([self._warmup_page])
                self._seq_current_page.append(self._warmup_page)
                self._seq_tokens_used.append(0)          # start empty
            else:
                self._seq_allocations.append([])
                self._seq_current_page.append(None)
                self._seq_tokens_used.append(0)

            extra_pages = self._allocate_tokens_for_seq(len(seq), i)
            self._seq_allocations[i].extend(extra_pages)

        return self._seq_allocations

    def step(self, num_seqs: int) -> List[List[Dict]]:
        """Append **one token** to each sequence."""
        if not self._seq_allocations:
            raise RuntimeError("Call warmup() / prefill() first")

        for i in range(num_seqs):
            extra = self._allocate_tokens_for_seq(1, i)
            # duplicate-safe because _allocate_tokens_for_seq appends each page once
            self._seq_allocations[i].extend(extra)

        return self._seq_allocations

    # ---------- diagnostics --------------------------------------------------
    def workers_used(self) -> List[int]:
        return sorted({wid for wid in self.table_id_to_worker.values()})

    def table_summary(self) -> Dict[int, int]:
        """Return {table_id: pages_per_table} for all realised tables."""
        return {tbl_id: self.pages_per_table for tbl_id in self.table_id_to_worker}

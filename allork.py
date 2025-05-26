from typing import List, Tuple, Dict, Callable, Optional

class PageBudgetExceeded(Exception):
    pass

class PageAllocatorStreamed:
    def __init__(
        self,
        page_size: int,
        pages_per_table: int,
        table_allocation_order: List[Tuple[int, int]],
        alloc_table: Callable[[int], int],
    ):
        self.page_size = page_size
        self.pages_per_table = pages_per_table
        self.alloc_table = alloc_table
        self.table_allocation_order = table_allocation_order

        self._worker_idx = 0
        self._table_idx = 0
        self._page_idx = 0

        self._tables: Dict[Tuple[int, int], int] = {}
        self.table_id_to_worker = {}

        self._warmup_page = None
        self._seq_allocations: List[List[Dict]] = []
        self._seq_current_pages: List[Optional[Dict]] = []
        self._seq_tokens_used_in_page: List[int] = []

    def _next_page(self) -> Dict:
        while self._worker_idx < len(self.table_allocation_order):
            wid, num_tables = self.table_allocation_order[self._worker_idx]
            if self._table_idx >= num_tables:
                self._worker_idx += 1
                self._table_idx = 0
                self._page_idx = 0
                continue

            if self._page_idx >= self.pages_per_table:
                self._table_idx += 1
                self._page_idx = 0
                continue

            key = (wid, self._table_idx)
            if key not in self._tables:
                table_id = self.alloc_table(wid)
                self._tables[key] = table_id
                self.table_id_to_worker[table_id] = wid

            table_id = self._tables[key]
            page_index = self._page_idx
            self._page_idx += 1
            return {"table": table_id, "page": page_index}

        raise PageBudgetExceeded("No more pages available from configured table allocation.")

    def _alloc_tokens_for_seq(self, num_tokens: int, seq_index: int) -> List[Dict]:
        pages = []
        while num_tokens > 0:
            current_page = self._seq_current_pages[seq_index]
            tokens_used = self._seq_tokens_used_in_page[seq_index]

            if current_page is None or tokens_used == self.page_size:
                current_page = self._next_page()
                self._seq_current_pages[seq_index] = current_page
                self._seq_tokens_used_in_page[seq_index] = 0
                tokens_used = 0

            page = current_page
            space_left = self.page_size - tokens_used
            use_now = min(num_tokens, space_left)
            num_tokens -= use_now
            self._seq_tokens_used_in_page[seq_index] += use_now
            pages.append(page)

        return pages

    def warmup(self) -> List[List[Dict]]:
        if self._warmup_page is not None:
            return self._seq_allocations
        self._warmup_page = self._next_page()
        self._seq_allocations = [[self._warmup_page]]
        self._seq_current_pages = [self._warmup_page]
        self._seq_tokens_used_in_page = [1]
        return self._seq_allocations

    def prefill(self, sequences: List[List[int]]) -> List[List[Dict]]:
        self._seq_allocations = []
        self._seq_current_pages = []
        self._seq_tokens_used_in_page = []

        for i, seq in enumerate(sequences):
            self._seq_current_pages.append(None)
            self._seq_tokens_used_in_page.append(0)
            pages = self._alloc_tokens_for_seq(len(seq), i)
            self._seq_allocations.append(pages)

        return self._seq_allocations

    def step(self, num_seqs: int) -> List[List[Dict]]:
        if not self._seq_allocations:
            raise RuntimeError("Must call prefill() or warmup() before step()")

        for i in range(num_seqs):
            pages = self._alloc_tokens_for_seq(1, i)
            self._seq_allocations[i].extend(pages)
        return self._seq_allocations

    def workers_used(self) -> List[int]:
        return sorted(set(self.table_id_to_worker.values()))

    def table_summary(self) -> Dict[int, int]:
        from collections import defaultdict
        usage = defaultdict(int)
        for (wid, tidx), tid in self._tables.items():
            usage[tid] = self.pages_per_table
        return dict(usage)



from typing import List, Tuple, Dict, Callable, Optional

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

        self._current_page: Optional[Dict] = None
        self._tokens_used_in_page = 0

        self._warmup_page = None
        self._seq_allocations: List[List[Dict]] = []  # Holds the full allocation per sequence

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

        raise RuntimeError("No more pages available")

    def _alloc_tokens(self, num_tokens: int) -> List[Dict]:
        result = []
        tokens_remaining = num_tokens

        while tokens_remaining > 0:
            if self._current_page is None or self._tokens_used_in_page == self.page_size:
                self._current_page = self._next_page()
                self._tokens_used_in_page = 0

            page = self._current_page
            space_left = self.page_size - self._tokens_used_in_page
            use_now = min(space_left, tokens_remaining)
            tokens_remaining -= use_now
            self._tokens_used_in_page += use_now
            result.append(page)

        return result

    def warmup(self) -> List[List[Dict]]:
        if self._warmup_page is not None:
            return self._seq_allocations
        self._warmup_page = self._next_page()
        self._seq_allocations = [[self._warmup_page]]
        return self._seq_allocations

    def prefill(self, seq_lens: List[List[int]]) -> List[List[Dict]]:
        self._seq_allocations = []
        for seq in seq_lens:
            pages = self._alloc_tokens(len(seq))
            self._seq_allocations.append(pages)
        return self._seq_allocations

    def step(self, num_seqs: int) -> List[List[Dict]]:
        if not self._seq_allocations:
            raise RuntimeError("Must call prefill() or warmup() before step()")
        for i in range(num_seqs):
            new_pages = self._alloc_tokens(1)
            self._seq_allocations[i].extend(new_pages)
        return self._seq_allocations

    def workers_used(self) -> List[int]:
        return sorted(set(self.table_id_to_worker.values()))

    def table_summary(self) -> Dict[int, int]:
        from collections import defaultdict
        usage = defaultdict(int)
        for (wid, tidx), tid in self._tables.items():
            usage[tid] = self.pages_per_table
        return dict(usage)

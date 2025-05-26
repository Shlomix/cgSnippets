from typing import List, Tuple, Dict, Callable, Optional
import math

class PageAllocatorStreamed:
    def __init__(
        self,
        pages_per_table: int,
        table_allocation_order: List[Tuple[int, int]],
        alloc_table: Callable[[int], int],
    ):
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
        self._page_size = None

        self._warmup_page = None

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

    def _alloc_tokens(self, num_tokens: int, page_size: int) -> List[Dict]:
        if self._page_size is None:
            self._page_size = page_size

        result = []
        tokens_remaining = num_tokens

        while tokens_remaining > 0:
            if self._current_page is None or self._tokens_used_in_page == page_size:
                self._current_page = self._next_page()
                self._tokens_used_in_page = 0

            page = self._current_page
            space_left = page_size - self._tokens_used_in_page
            use_now = min(space_left, tokens_remaining)
            tokens_remaining -= use_now
            self._tokens_used_in_page += use_now
            result.append(page)

        return result

    def warmup(self) -> Dict:
        if self._warmup_page is not None:
            return self._warmup_page
        self._warmup_page = self._next_page()
        return self._warmup_page

    def prefill(self, seq_lens: List[List[int]], page_size: int) -> List[List[Dict]]:
        output = []
        for seq in seq_lens:
            pages = self._alloc_tokens(len(seq), page_size)
            output.append(pages)
        return output

    def step(self, num_seqs: int, page_size: int) -> List[Dict]:
        return self._alloc_tokens(num_seqs, page_size)

    def workers_used(self) -> List[int]:
        return sorted(set(self.table_id_to_worker.values()))

    def table_summary(self) -> Dict[int, int]:
        from collections import defaultdict
        usage = defaultdict(int)
        for (wid, tidx), tid in self._tables.items():
            usage[tid] = self.pages_per_table
        return dict(usage)
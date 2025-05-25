from typing import List, Dict, Callable, Optional
import math

class PageAllocator:
    def __init__(self,
        page_size: int,
        pages_per_table: int,
        remote_worker_ids: List[int],
        decode_tokens: int,
        alloc_table: Callable[[int], int],
        locals_present: bool = True,
        allow_single_local: bool = False,
        expandable_remote: Optional[int] = None,
    ):
        self.page_size = page_size
        self.pages_per_table = pages_per_table
        self.remote_worker_ids = remote_worker_ids
        self.decode_tokens = decode_tokens
        self.alloc_table = alloc_table
        self.locals_present = locals_present
        self.allow_single_local = allow_single_local
        self.expandable_remote = expandable_remote if expandable_remote is not None else (remote_worker_ids[-1] if remote_worker_ids else None)
        self.local_tables = []
        self.remote_tables = {wid: [] for wid in remote_worker_ids}
        self.table_id_to_worker = {}
        self._warmup_page = None

    def step(self, num_seqs: int) -> List[Dict]:
        total_tokens = num_seqs  # 1 token per sequence
        total_pages = math.ceil(total_tokens / self.page_size)
        return self._alloc_pages(total_pages, spill_to_remotes=True)

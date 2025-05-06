# ---------------------------------------------------------------------------
#  Dummy stand‑in for your fem_boost_model
# ---------------------------------------------------------------------------
class DummyBoostModel:
    """Returns monotonically‑increasing table IDs."""
    def __init__(self, start_id: int = 0):
        self._next_id = start_id

    def alloc_table(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid


# ---------------------------------------------------------------------------
#  SimplePagedAllocator (worker‑first, fill‑then‑move)
# ---------------------------------------------------------------------------
import math
from collections import defaultdict
from typing import Dict, List, Tuple


class SimplePagedAllocator:
    """
    Page allocator for **one batch** of sequences, filling worker 0 first.

    Parameters
    ----------
    input_ids            : list[list[int]]
    worker_table_counts  : Dict[int, int]   # {worker_id: num_tables}
    fem_boost_model      : object with .alloc_table()
    page_size            : int   # tokens per page
    table_capacity       : int   # pages a table can hold
    """

    def __init__(self,
                 input_ids: List[List[int]],
                 worker_table_counts: Dict[int, int],
                 fem_boost_model,
                 page_size: int,
                 table_capacity: int):

        self.input_ids      = input_ids
        self.N              = len(input_ids)
        self.page_size      = page_size
        self.table_capacity = table_capacity

        # ---- flat list of (worker_id, table_id) in ASCENDING worker order
        self.tables: List[Tuple[int, int]] = [
            (wid, fem_boost_model.alloc_table())
            for wid in sorted(worker_table_counts)          # deterministic
            for _  in range(worker_table_counts[wid])
        ]
        if not self.tables:
            raise ValueError("Allocator created with zero tables.")

        # bookkeeping structures
        self._next_page_id = defaultdict(int)               # table → next idx
        self._used_pages   = defaultdict(int)               # table → pages used
        self._table_idx    = 0                              # current table ptr
        self._seq_tokens   = [0] * self.N                   # token counter
        self.page_mappings: List[List[dict]] = [[] for _ in range(self.N)]

        # allocate prompt pages immediately
        for seq_id, tokens in enumerate(map(len, input_ids)):
            self._allocate_pages(seq_id, tokens)

    # ------------------------------------------------------------------ #
    # public: +1 token for every sequence                                #
    # ------------------------------------------------------------------ #
    def step(self) -> List[List[dict]]:
        for seq_id in range(self.N):
            prev = self._seq_tokens[seq_id]
            self._seq_tokens[seq_id] = prev + 1
            if prev % self.page_size == 0:                  # new page needed
                self._allocate_pages(seq_id, 0, force_one=True)
        return self.page_mappings

    # ------------------------------------------------------------------ #
    # internal helper                                                    #
    # ------------------------------------------------------------------ #
    def _allocate_pages(self,
                        seq_id: int,
                        n_tokens: int,
                        force_one: bool = False):
        pages = 1 if force_one else math.ceil(n_tokens / self.page_size)

        for _ in range(pages):
            worker_id, table_id = self.tables[self._table_idx]
            page_id             = self._next_page_id[table_id]

            self.page_mappings[seq_id].append(
                {"worker_id": worker_id,
                 "table_id":  table_id,
                 "page_id":   page_id}
            )

            # bookkeeping
            self._next_page_id[table_id] += 1
            self._used_pages[table_id]   += 1

            # move to *next table* only when current one is full
            if self._used_pages[table_id] >= self.table_capacity:
                self._table_idx = (self._table_idx + 1) % len(self.tables)

        # update overall token counter (skip when force_one)
        self._seq_tokens[seq_id] += n_tokens

if __name__ == "__main__":
    batch = [[1, 2, 3]] * 5                    # 5 sequences
    workers = {0: 2, 1: 0, 2: 3}               # worker‑0 first

    alloc = SimplePagedAllocator(batch, workers,
                                 DummyBoostModel(),
                                 page_size=4,
                                 table_capacity=6)

    for i, m in enumerate(alloc.page_mappings):
        print(f"seq{i}: {m}")

from math import ceil
from typing import Dict, List, Optional

LOCAL = -1  # id for the local worker


class DummyBoostModel:
    """Stand‑in for fem_boost_model; hands out unique table IDs."""
    def __init__(self, start: int = 0):
        self._next = start
        self.calls = []                       # list[(where, table_id)]

    def alloc_table(self, where: int) -> int:
        tid, self._next = self._next, self._next + 1
        self.calls.append((where, tid))
        return tid


class LazyBatchAllocator:
    """Single‑batch, fill‑current‑worker‑first, tables created lazily."""

    def __init__(
        self,
        input_ids: List[List[int]],
        worker_quota: Dict[int, int],             # must include LOCAL
        fem_boost_model,
        page_size: int,
        table_capacity: int,
        *,
        preallocated: Optional[Dict[int, List[int]]] = None,
    ):
        if LOCAL not in worker_quota:
            raise ValueError("worker_quota must include -1 (local)")
        if not input_ids:
            raise ValueError("empty batch")

        self.psize = page_size
        self.cap   = table_capacity
        self.model = fem_boost_model
        self.max_q = worker_quota

        self.workers = (
            [LOCAL] if worker_quota[LOCAL] > 0 else []
        ) + [w for w in sorted(worker_quota) if w >= 0 and worker_quota[w] > 0]
        if not self.workers:
            raise ValueError("all worker quotas are zero")

        self.tables: Dict[int, List[int]] = {w: [] for w in self.workers}
        self.used:   Dict[int, List[int]] = {w: [] for w in self.workers}
        self.pid:    Dict[int, List[int]] = {w: [] for w in self.workers}

        if preallocated:
            for w, tids in preallocated.items():
                if w not in self.tables:
                    raise ValueError(f"worker {w} has zero quota")
                for t in tids:
                    self._attach(w, t)

        self.w_ptr   = 0
        self.tokens  = [len(s) for s in input_ids]
        self.mapping = [[] for _ in input_ids]          # exposed to user

        for seq, tok in enumerate(self.tokens):
            for _ in range(ceil(tok / self.psize)):
                self._alloc_page(seq)

    # ------------------------------------------------------------------
    def step(self):
        for seq in range(len(self.tokens)):
            if self.tokens[seq] % self.psize == 0:
                self._alloc_page(seq)
            self.tokens[seq] += 1
        return self.mapping

    # ------------------------------------------------------------------
    def _alloc_page(self, seq: int):
        nW = len(self.workers)
        for off in range(nW):
            w = self.workers[(self.w_ptr + off) % nW]

            # existing table with space?
            for idx, tid in enumerate(self.tables[w]):
                if self.used[w][idx] < self.cap:
                    self._place(seq, w, idx, tid)
                    self.w_ptr = (self.w_ptr + off) % nW
                    return

            # new table if quota allows
            if len(self.tables[w]) < self.max_q[w]:
                tid = self.model.alloc_table(w)
                self._attach(w, tid)
                idx = len(self.tables[w]) - 1
                self._place(seq, w, idx, tid)
                self.w_ptr = (self.w_ptr + off) % nW
                return
        raise RuntimeError("capacity exhausted")

    # ------------------------------------------------------------------
    def _attach(self, wid: int, tid: int):
        self.tables[wid].append(tid)
        self.used[wid].append(0)
        self.pid[wid].append(0)

    def _place(self, seq: int, wid: int, idx: int, tid: int):
        self.mapping[seq].append({"table_id": tid, "page_id": self.pid[wid][idx]})
        self.pid[wid][idx] += 1
        self.used[wid][idx] += 1

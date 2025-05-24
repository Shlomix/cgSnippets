# page_allocator.py – compact Python 3.9 version
import math
from typing import List, Dict, Callable

class SimpleTableFactory:
    def __init__(self) -> None:
        self._n = 0
    def __call__(self, _: bool) -> int:
        self._n, tid = self._n + 1, self._n
        return tid

class PageAllocator:
    def __init__(
        self,
        *,
        page_size: int,
        pages_per_table: int,
        remote_worker_ids: List[int],
        decode_tokens: int,
        table_factory: Callable[[bool], int],
        locals_present: bool = True,
        allow_single_local: bool = False,
    ) -> None:
        if not locals_present and not remote_worker_ids:
            raise ValueError("No locals and no remotes")
        self.P, self.T = page_size, pages_per_table
        self.RW, self.D = remote_worker_ids, decode_tokens
        self.locals_present, self.single_ok = locals_present, allow_single_local
        self.new_tid = table_factory
        self.tables: Dict[int, Dict] = {}
        self.locals: List[int] = []
        self.remotes: List[int] = []
        self.cursor: List[int] = []
        self.assign = 0
        self.seq_pages: List[List[Dict]] = []
        self.N = 0
        self._warm = None

    # ───────────────── helpers ─────────────────
    def _plan(self, seq_lens: List[int]):
        self.N = len(seq_lens)
        pre = sum(math.ceil(l / self.P) for l in seq_lens)
        dec = 0 if self.D == 0 else self.N * math.ceil(self.D / self.P)
        tot = pre + dec
        loc = 0
        if self.locals_present:
            loc = max(1, math.ceil(pre / self.T))
            if not self.RW:
                loc = math.ceil(tot / self.T)
                if not self.single_ok:
                    loc = max(loc, 2)
        rem_pages = max(0, tot - loc * self.T)
        rem_tables = 0
        if self.RW:
            rem_tables = max(len(self.RW), math.ceil(rem_pages / self.T))
            if rem_pages < rem_tables * self.T:
                raise ValueError("decode under‑utilises remotes")
        elif rem_pages:
            raise ValueError("remotes needed but none provided")
        self._mk_tables(loc, rem_tables)

    def _mk_tables(self, loc: int, rem: int):
        while len(self.locals) < loc:
            tid = self.new_tid(False)
            self.locals.append(tid)
            self.tables[tid] = {"pages": 0, "worker": -1}
        while len(self.remotes) < rem:
            tid = self.new_tid(True)
            wid = self.RW[self.assign % len(self.RW)]
            self.assign += 1
            self.remotes.append(tid)
            self.tables[tid] = {"pages": 0, "worker": wid}
        self.cursor = self.locals + self.remotes

    def _page(self) -> Dict:
        for tid in self.cursor:
            meta = self.tables[tid]
            if meta["pages"] < self.T:
                pid = meta["pages"]
                meta["pages"] += 1
                return {"table_id": tid, "page_id": pid}
        raise RuntimeError("out of tables")

    # ───────────────── API ─────────────────────
    def warmup(self):
        if self._warm is None:
            if not self.tables:
                self._plan([])
            self._warm = {"table_id": self.cursor[0], "page_id": 0}
        return self._warm

    def prefill(self, seq_lens: List[int]):
        if not self.tables:
            self._plan(seq_lens)
            self.seq_pages = [[] for _ in range(self.N)]
        for i, tok in enumerate(seq_lens):
            for _ in range(math.ceil(tok / self.P)):
                self.seq_pages[i].append(self._page())
        return self.seq_pages

    def decode(self):
        self.cursor = self.locals + self.remotes
        each = math.ceil(self.D / self.P)
        for i in range(self.N):
            for _ in range(each):
                self.seq_pages[i].append(self._page())
        return self.seq_pages

    # ───── introspection for tests ─────
    def workers_used(self):
        return sorted({m["worker"] for m in self.tables.values() if m["pages"]})


# ───────────── tests ─────────────
SEQ_LENS = [16, 16, 16, 16]
PAGE_SIZE = 16
PAGES_PER_TABLE = 2
DECODE_OPTS = [1, 4, 64]

def run_case(rem, loc, single, dec):
    try:
        alloc = PageAllocator(
            page_size=PAGE_SIZE,
            pages_per_table=PAGES_PER_TABLE,
            remote_worker_ids=list(range(rem)),
            decode_tokens=dec,
            table_factory=SimpleTableFactory(),
            locals_present=loc,
            allow_single_local=single,
        )
    except ValueError as e:
        print(f"rem={rem} loc={loc} dec={dec} SKIP → {e}")
        return
    alloc.prefill(SEQ_LENS)
    alloc.decode()
    print(f"rem={rem} loc={loc} dec={dec} workers {alloc.workers_used()}")

def run_all():
    for loc in (True, False):
        for single in (False, True):
            for rem in range(4):
                for dec in DECODE_OPTS:
                    run_case(rem, loc, single, dec)
    run_case(2, True, False, 48)
    print("✅")

if __name__ == "__main__":
    run_all()

from typing import List, Dict, Callable, Optional
import math

class PageAllocator:
    def __init__(
        self,
        page_size: int,
        pages_per_table: int,
        remote_worker_ids: List[int],
        decode_tokens: int,
        alloc_table: Callable[[int], int],
        locals_present: bool = True,
        allow_single_local: bool = False,
        expandable_remote: Optional[int] = None,
    ):
        if not locals_present and not remote_worker_ids:
            raise ValueError("No locals and no remotes")

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

    def warmup(self) -> Dict:
        if self._warmup_page is not None:
            return self._warmup_page

        if self.locals_present:
            table_id = self.alloc_table(-1)
            self.local_tables.append((table_id, []))
            self.table_id_to_worker[table_id] = -1
            page_id = 0
            self._warmup_page = {"table": table_id, "page": page_id}
            return self._warmup_page
        elif self.remote_worker_ids:
            first_remote = self.remote_worker_ids[0]
            table_id = self.alloc_table(first_remote)
            self.remote_tables[first_remote].append((table_id, []))
            self.table_id_to_worker[table_id] = first_remote
            self._warmup_page = {"table": table_id, "page": 0}
            return self._warmup_page
        else:
            raise RuntimeError("No available workers for warmup")

    def _alloc_pages(self, total_pages: int, spill_to_remotes: bool) -> List[Dict]:
        pages = []
        table_cursor = []

        local_table_cap = 1 if self.allow_single_local else 2
        if self.locals_present:
            while len(self.local_tables) < local_table_cap:
                table_id = self.alloc_table(-1)
                self.local_tables.append((table_id, []))
                self.table_id_to_worker[table_id] = -1

        for table_id, _ in self.local_tables:
            table_cursor.append((table_id, -1))

        if self.remote_worker_ids and (spill_to_remotes or not self.locals_present):
            for wid in self.remote_worker_ids:
                if not self.remote_tables[wid]:
                    table_id = self.alloc_table(wid)
                    self.remote_tables[wid].append((table_id, []))
                    self.table_id_to_worker[table_id] = wid
                for table_id, _ in self.remote_tables[wid]:
                    table_cursor.append((table_id, wid))

        cursor_index = 0
        visited = set()

        while len(pages) < total_pages:
            if not table_cursor:
                raise RuntimeError("No tables available to allocate pages")

            if len(visited) >= len(table_cursor):
                raise RuntimeError("Cursor is stuck — possible allocation deadlock")

            table_id, worker_id = table_cursor[cursor_index]
            cursor_index = (cursor_index + 1) % len(table_cursor)
            key = (table_id, worker_id)
            if key in visited:
                continue
            visited.add(key)

            table_list = self.local_tables if worker_id == -1 else self.remote_tables[worker_id]
            entry = [x for x in table_list if x[0] == table_id]
            if not entry:
                continue

            current_table = entry[0][1]

            if len(current_table) >= self.pages_per_table:
                if worker_id == -1:
                    if not self.allow_single_local or not self.remote_worker_ids:
                        table_id = self.alloc_table(-1)
                        new_table = (table_id, [])
                        self.local_tables.append(new_table)
                        self.table_id_to_worker[table_id] = -1
                        table_cursor.append((table_id, -1))
                        visited.clear()
                    continue
                elif spill_to_remotes or not self.locals_present:
                    if worker_id == self.expandable_remote:
                        table_id = self.alloc_table(worker_id)
                        new_table = (table_id, [])
                        self.remote_tables[worker_id].append(new_table)
                        self.table_id_to_worker[table_id] = worker_id
                        table_cursor.append((table_id, worker_id))
                        visited.clear()
                    continue
                continue

            page_index = len(current_table)
            current_table.append(None)
            pages.append({"table": table_id, "page": page_index})
            visited.clear()

        return pages

    def prefill(self, seq_lens: List[int]) -> List[List[Dict]]:
        total_tokens = sum(seq_lens)
        total_pages = math.ceil(total_tokens / self.page_size)
        spill = self.allow_single_local and bool(self.remote_worker_ids)
        all_pages = self._alloc_pages(total_pages, spill_to_remotes=spill)

        output = []
        index = 0
        for seq_len in seq_lens:
            seq_pages = math.ceil(seq_len / self.page_size)
            output.append(all_pages[index: index + seq_pages])
            index += seq_pages
        return output

    def decode(self) -> List[List[Dict]]:
        num_seqs = 4
        total_tokens = num_seqs * self.decode_tokens
        total_pages = math.ceil(total_tokens / self.page_size)
        pages = self._alloc_pages(total_pages, spill_to_remotes=True)
        pages_per_seq = len(pages) // num_seqs
        return [pages[i * pages_per_seq: (i + 1) * pages_per_seq] for i in range(num_seqs)]

    def workers_used(self) -> List[int]:
        return sorted(set(self.table_id_to_worker.values()))

    def table_summary(self) -> Dict[int, int]:
        summary = {}
        for table_list in self.local_tables:
            summary[table_list[0]] = len(table_list[1])
        for wlist in self.remote_tables.values():
            for table_id, pages in wlist:
                summary[table_id] = len(pages)
        return summary

# ============== TEST RUNNER ==============

def run_one_case(remote_ids, locals_present, allow_single_local, expandable_remote=None):
    table_log = []
    def alloc_table(worker_id):
        table_id = len(table_log)
        table_log.append((worker_id, table_id))
        return table_id

    PAGE_SIZE = 16
    PAGES_PER_TABLE = 2
    DECODE_TOKENS = 4
    SEQ_LENS = [16, 16, 16, 16]

    try:
        allocator = PageAllocator(
            page_size=PAGE_SIZE,
            pages_per_table=PAGES_PER_TABLE,
            remote_worker_ids=remote_ids,
            decode_tokens=DECODE_TOKENS,
            alloc_table=alloc_table,
            locals_present=locals_present,
            allow_single_local=allow_single_local,
            expandable_remote=expandable_remote,
        )

        warmup_page = allocator.warmup()
        prefill_result = allocator.prefill(SEQ_LENS)
        decode_result = allocator.decode()
        summary = allocator.table_summary()
        workers = allocator.workers_used()

        print("✅ PASS", {
            "remotes": len(remote_ids),
            "locals": int(locals_present),
            "allow_single_local": allow_single_local,
            "expandable_remote": expandable_remote,
            "warmup": warmup_page,
            "prefill pages": sum(len(p) for p in prefill_result),
            "decode pages": sum(len(p) for p in decode_result),
            "total_tables": len(summary),
            "workers_used": workers,
            "table_log": table_log
        })
    except Exception as e:
        print("❌ FAIL", {
            "remotes": len(remote_ids),
            "locals": int(locals_present),
            "allow_single_local": allow_single_local,
            "expandable_remote": expandable_remote,
            "error": str(e)
        })

def run_sanity_case():
    table_log = []
    def alloc_table(worker_id):
        table_id = len(table_log)
        table_log.append((worker_id, table_id))
        return table_id

    PAGE_SIZE = 16
    PAGES_PER_TABLE = 1536
    DECODE_TOKENS = 4
    SEQ_LENS = [16] * 64

    try:
        allocator = PageAllocator(
            page_size=PAGE_SIZE,
            pages_per_table=PAGES_PER_TABLE,
            remote_worker_ids=[],
            decode_tokens=DECODE_TOKENS,
            alloc_table=alloc_table,
            locals_present=True,
            allow_single_local=True,
        )

        warmup_page = allocator.warmup()
        prefill_result = allocator.prefill(SEQ_LENS)
        decode_result = allocator.decode()
        summary = allocator.table_summary()
        workers = allocator.workers_used()

        assert len(summary) == 1 and -1 in workers, "Expected exactly one local table"

        print("🧪 SANITY PASS", {
            "warmup": warmup_page,
            "prefill pages": sum(len(p) for p in prefill_result),
            "decode pages": sum(len(p) for p in decode_result),
            "total_tables": len(summary),
            "workers_used": workers,
            "table_log": table_log
        })
    except Exception as e:
        print("🧪 SANITY FAIL", {"error": str(e)})

if __name__ == '__main__':
    for num_remotes in range(4):
        remote_ids = list(range(num_remotes))
        for locals_present in (True, False):
            for allow_single in (False, True):
                for expandable in remote_ids if remote_ids else [None]:
                    run_one_case(remote_ids, locals_present, allow_single, expandable_remote=expandable)
    run_sanity_case()

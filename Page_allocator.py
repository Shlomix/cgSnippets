import random
from math import ceil
from typing import Dict, List, Tuple, Set


def create_page_mappings(
    input_ids: List[List[int]],
    fem_boost_model,
    *,
    page_size: int,
    future_tokens: int,
    pages_per_table: int,
    tables_per_worker: Dict[int, int] | List[Tuple[int, int]],
    random_table_assignment: bool = False,
    random_seed: int | None = None,
) -> List[List[Dict[str, int]]]:
    """
    Allocates pages for every sequence and returns
        List[List[{'table_id': int, 'page_id': int}]]  (aligned with input_ids).

    Extra guarantees:
      • *All* tables described in `tables_per_worker` are allocated at startup.
        If some end up unused a message is printed.
      • No (table_id, page_id) pair is ever given to more than one sequence.
    """

    # ------------------------------------------------------------------ #
    # 1.  Allocate every table we might use
    # ------------------------------------------------------------------ #
    if isinstance(tables_per_worker, dict):
        worker_items = list(tables_per_worker.items())
    else:
        worker_items = list(tables_per_worker)

    if random_table_assignment:
        random.Random(random_seed).shuffle(worker_items)

    table_pool: List[Dict[str, int]] = []       # {worker_id, table_id, used}
    for worker_id, n in worker_items:
        for _ in range(n):
            tid = fem_boost_model.alloc_table(worker_id=worker_id)
            table_pool.append({'worker_id': worker_id,
                               'table_id': tid,
                               'used': 0})

    if not table_pool:
        raise ValueError("tables_per_worker supplied no tables")

    # iterator over tables in the fixed order we created them
    tbl_idx = 0
    cur_tbl = table_pool[0]

    # ------------------------------------------------------------------ #
    # 2.  Allocate pages sequence by sequence
    # ------------------------------------------------------------------ #
    seen_pages: Set[Tuple[int, int]] = set()          # for uniqueness check
    page_mappings: List[List[Dict[str, int]]] = []

    for seq in input_ids:
        pages_needed = ceil((len(seq) + future_tokens) / page_size)
        seq_map: List[Dict[str, int]] = []

        while pages_needed > 0:
            free_slots = pages_per_table - cur_tbl['used']
            if free_slots == 0:                       # table full → next table
                tbl_idx += 1
                if tbl_idx >= len(table_pool):
                    raise RuntimeError(
                        "Ran out of table capacity: add more tables or raise "
                        "`pages_per_table`."
                    )
                cur_tbl = table_pool[tbl_idx]
                continue

            take = min(free_slots, pages_needed)
            base = cur_tbl['used']
            for off in range(take):
                page = {'table_id': cur_tbl['table_id'], 'page_id': base + off}
                key = (page['table_id'], page['page_id'])
                assert key not in seen_pages, (
                    f"Duplicate page assignment detected for {key}"
                )
                seen_pages.add(key)
                seq_map.append(page)

            cur_tbl['used'] += take
            pages_needed    -= take

        page_mappings.append(seq_map)

    # ------------------------------------------------------------------ #
    # 3.  Warn if some tables stayed empty
    # ------------------------------------------------------------------ #
    unused = [t['table_id'] for t in table_pool if t['used'] == 0]
    if unused:
        print(f"[allocator notice] {len(unused)} table(s) never used: {unused}")

    return page_mappings

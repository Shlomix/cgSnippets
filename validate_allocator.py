import json
from pathlib import Path
from page_allocator import PageAllocator

json_dir = Path("allocator_json_step_tests")
json_files = list(json_dir.glob("*.json"))

for json_file in json_files:
    with open(json_file) as f:
        cfg = json.load(f)

    table_log = []
    def alloc_table(worker_id: int) -> int:
        table_id = len(table_log)
        table_log.append((worker_id, table_id))
        return table_id

    allocator = PageAllocator(
        page_size=cfg["page_size"],
        pages_per_table=cfg["pages_per_table"],
        remote_worker_ids=cfg["remote_worker_ids"],
        decode_tokens=cfg["decode_tokens"],
        alloc_table=alloc_table,
        locals_present=cfg["locals_present"],
        allow_single_local=cfg["allow_single_local"],
        expandable_remote=cfg["expandable_remote"],
    )

    allocator.warmup()
    seq_lens = [len(x) for x in cfg["input_ids"]]
    allocator.prefill(seq_lens)

    for _ in range(cfg["decode_tokens"]):
        allocator.step(len(cfg["input_ids"]))

    print({
        "test": json_file.name,
        "workers_used": allocator.workers_used(),
        "total_tables": len(allocator.table_summary()),
        "table_log": table_log
    })

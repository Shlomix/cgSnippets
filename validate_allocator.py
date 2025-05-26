import json
import math
import os
from page_allocator_streamed import PageAllocatorStreamed

def alloc_table_logger(log):
    def allocator(worker_id):
        tid = len(log)
        log.append((worker_id, tid))
        return tid
    return allocator

results = []

for filename in os.listdir("test_jsons"):
    if not filename.endswith(".json"):
        continue
    with open(os.path.join("test_jsons", filename)) as f:
        cfg = json.load(f)

    log = []
    allocator = PageAllocatorStreamed(
        pages_per_table=cfg["pages_per_table"],
        table_allocation_order=cfg["table_allocation_order"],
        alloc_table=alloc_table_logger(log)
    )

    try:
        allocator.warmup()
        allocator.prefill(cfg["input_ids"], cfg["page_size"])
        for _ in range(cfg["decode_tokens"]):
            allocator.step(len(cfg["input_ids"]), cfg["page_size"])

        results.append((filename, "PASS", log))
    except Exception as e:
        results.append((filename, "FAIL", str(e)))

for name, status, details in results:
    print(f"{name:<30} {status} {details if status == 'FAIL' else ''}")
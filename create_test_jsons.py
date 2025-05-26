import os
import json
import math

def suggest_table_allocation_order(num_tables, remote_worker_ids, last_remote_grows=True):
    if not remote_worker_ids:
        return [(-1, num_tables)]

    order = [(-1, 1)]
    rem_tables = max(0, num_tables - 1)

    if last_remote_grows:
        base = rem_tables // len(remote_worker_ids)
        remainder = rem_tables % len(remote_worker_ids)
        for i, wid in enumerate(remote_worker_ids):
            extra = remainder if i == len(remote_worker_ids) - 1 else 0
            order.append((wid, base + extra))
    else:
        per = math.ceil(rem_tables / len(remote_worker_ids))
        for wid in remote_worker_ids:
            order.append((wid, per))

    return order

test_cases = {
    "local_only": {
        "input_ids": [list(range(16))] * 4,
        "decode_tokens": 4,
        "page_size": 16,
        "pages_per_table": 2,
        "remote_worker_ids": []
    },
    "local_and_1_remote": {
        "input_ids": [list(range(16))] * 4,
        "decode_tokens": 4,
        "page_size": 16,
        "pages_per_table": 2,
        "remote_worker_ids": [0]
    },
    "local_and_2_remotes": {
        "input_ids": [list(range(16))] * 4,
        "decode_tokens": 4,
        "page_size": 16,
        "pages_per_table": 2,
        "remote_worker_ids": [0, 1]
    },
    "remotes_only_2": {
        "input_ids": [list(range(16))] * 4,
        "decode_tokens": 4,
        "page_size": 16,
        "pages_per_table": 2,
        "remote_worker_ids": [0, 1]
    },
    "sanity_local_bigtable": {
        "input_ids": [list(range(16))] * 64,
        "decode_tokens": 8,
        "page_size": 16,
        "pages_per_table": 1536,
        "remote_worker_ids": []
    }
}

os.makedirs("test_jsons", exist_ok=True)

for name, cfg in test_cases.items():
    total_tokens = sum(len(x) for x in cfg["input_ids"]) + cfg["decode_tokens"] * len(cfg["input_ids"])
    total_pages = math.ceil(total_tokens / cfg["page_size"])
    total_tables = math.ceil(total_pages / cfg["pages_per_table"])
    table_order = suggest_table_allocation_order(total_tables, cfg["remote_worker_ids"])
    cfg["table_allocation_order"] = table_order

    with open(f"test_jsons/{name}.json", "w") as f:
        json.dump(cfg, f, indent=2)
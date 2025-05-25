import json
import os

output_dir = "allocator_json_step_tests"
os.makedirs(output_dir, exist_ok=True)

PAGE_SIZE = 16
PAGES_PER_TABLE = 2
DECODE_TOKENS = 4
INPUT_IDS = [list(range(16))] * 4
EXPECTED_OUTPUT = "dummy_output"

configs = [
    {"name": "local_only", "remote_worker_ids": [], "locals_present": True, "allow_single_local": False},
    {"name": "local_and_1_remote", "remote_worker_ids": [0], "locals_present": True, "allow_single_local": False},
    {"name": "local_and_2_remotes", "remote_worker_ids": [0, 1], "locals_present": True, "allow_single_local": False},
    {"name": "remotes_only_2", "remote_worker_ids": [0, 1], "locals_present": False, "allow_single_local": False},
    {"name": "step_mode_2_remotes_only", "remote_worker_ids": [0, 1], "locals_present": False, "allow_single_local": False, "step_mode": True},
    {"name": "sanity_local_bigtable", "remote_worker_ids": [], "locals_present": True, "allow_single_local": True, "pages_per_table": 1536, "input_ids": [list(range(16))] * 64, "expected_output": "dummy_output"}
]

for cfg in configs:
    config = {
        "page_size": PAGE_SIZE,
        "pages_per_table": cfg.get("pages_per_table", PAGES_PER_TABLE),
        "decode_tokens": DECODE_TOKENS,
        "remote_worker_ids": cfg["remote_worker_ids"],
        "locals_present": cfg["locals_present"],
        "allow_single_local": cfg["allow_single_local"],
        "expandable_remote": cfg["remote_worker_ids"][-1] if cfg["remote_worker_ids"] else None,
        "step_mode": cfg.get("step_mode", False),
        "input_ids": cfg.get("input_ids", INPUT_IDS),
        "expected_output": cfg.get("expected_output", EXPECTED_OUTPUT)
    }

    with open(os.path.join(output_dir, f"{cfg['name']}.json"), "w") as f:
        json.dump(config, f, indent=2)

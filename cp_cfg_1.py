import json
import shutil
from pathlib import Path
from itertools import chain

# === Configuration ===
SRC_BASE = Path("/absolute/path/to/cfg_debugs").resolve(strict=True)  # <-- SET THIS!
DST_BASE = Path("/root/blue/tubes/boost_sdk/boost_tests_experimental/tests_blueprint")
CONFIG_PREFIX = Path("/root/blue/tubes/boost_sdk/boost_tests_experimental")

DRY_RUN = False      # Change to True to simulate only
OVERWRITE = False    # Change to True to allow overwriting files in destination

# === Gather Directories to Process ===
pattern_dirs = SRC_BASE.glob("BST*")
manual_dirs = [SRC_BASE / "F1", SRC_BASE / "F2"]
all_dirs = sorted(chain(pattern_dirs, manual_dirs))

# === Main Loop ===
for case_dir in all_dirs:
    case_name = case_dir.name  # e.g., BST1, F1
    config_json_path = case_dir / "config.json"
    config_yaml_path = case_dir / "config.yaml"

    if not case_dir.exists():
        print(f"[SKIP] Directory does not exist: {case_dir}")
        continue

    if not config_json_path.exists():
        print(f"[SKIP] Missing JSON: {config_json_path}")
        continue
    if not config_yaml_path.exists():
        print(f"[SKIP] Missing YAML: {config_yaml_path}")
        continue

    # Load and validate JSON
    try:
        with config_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read JSON {config_json_path}: {e}")
        continue

    try:
        old_cfg_path = Path(data["logical_blueprint"]["config_filename"])
        if not isinstance(data["logical_blueprint"]["config_filename"], str):
            raise TypeError("config_filename must be a string")
    except (KeyError, TypeError) as e:
        print(f"[ERROR] Invalid config_filename in {config_json_path}: {e}")
        continue

    new_cfg_filename = str(CONFIG_PREFIX / case_name / old_cfg_path.name)
    data["logical_blueprint"]["config_filename"] = new_cfg_filename

    # Prepare destination paths
    dst_dir = DST_BASE / case_name
    dst_json_path = dst_dir / "config.json"
    dst_yaml_path = dst_dir / "config.yaml"

    if not dst_dir.exists():
        print(f"[INFO] Creating directory: {dst_dir}")
        if not DRY_RUN:
            dst_dir.mkdir(parents=True, exist_ok=True)

    # Defensive overwrite check
    for fpath in [dst_json_path, dst_yaml_path]:
        if fpath.exists() and not OVERWRITE:
            print(f"[SKIP] File exists and overwrite is off: {fpath}")
            continue

    # Write updated config.json
    print(f"[WRITE] {dst_json_path}")
    if not DRY_RUN:
        try:
            with dst_json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to write JSON: {dst_json_path} — {e}")
            continue

    # Copy config.yaml
    print(f"[COPY] {config_yaml_path} → {dst_yaml_path}")
    if not DRY_RUN:
        try:
            shutil.copy2(config_yaml_path, dst_yaml_path)
        except Exception as e:
            print(f"[ERROR] Failed to copy YAML: {e}")
            continue

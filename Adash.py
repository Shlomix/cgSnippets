import json
import shutil
from pathlib import Path
from itertools import chain

# ──► EDIT THESE PATHS ◄──
SRC_BASE      = Path("/absolute/path/to/cfg_debugs").resolve(strict=True)
DST_BASE      = Path("/root/blue/tubes/boost_sdk/boost_tests_experimental/tests_blueprint")
CONFIG_PREFIX = Path("/root/blue/tubes/boost_sdk/boost_tests_experimental")
INFO_SRC_ROOT = Path("/absolute/path/to/info_files").resolve(strict=True)

# ──► SAFETY FLAGS ◄──
DRY_RUN   = False      # True = simulate only
OVERWRITE = False      # False = never overwrite existing dst files

# ────────────────────────────────────────────────────────────────────────────────
# 1. Build an index of the extra info files we need to copy → test_info.json
info_index = {}                           # {'BST3': Path(.../BST3.json), 'F2': Path(.../F2.json)}
for p in INFO_SRC_ROOT.rglob("*.json"):
    stem = p.stem
    if not (stem.startswith(("BST", "F")) and len(stem) <= 4):
        # Skip anything that is NOT a simple BSTx.json or Fx.json
        continue
    if stem not in info_index:
        info_index[stem] = p
    else:
        print(f"[WARN] duplicate info file for {stem}: {p} (already have {info_index[stem]})")

# 2. Collect every case directory we need to process
pattern_dirs = SRC_BASE.glob("BST*")
manual_dirs  = [SRC_BASE / "F1", SRC_BASE / "F2"]
all_dirs = sorted(chain(pattern_dirs, manual_dirs))

# 3. Helper for safe writes / copies
def safe_action(label: str, dst: Path, work):
    """Run work() unless DRY_RUN or overwrite is disallowed."""
    if dst.exists() and not OVERWRITE:
        print(f"[SKIP] {label} exists and OVERWRITE=False: {dst}")
        return
    print(f"[{label}] {dst}")
    if not DRY_RUN:
        work()

# 4. Main loop
for case_dir in all_dirs:
    if not case_dir.exists():
        print(f"[SKIP] Directory does not exist: {case_dir}")
        continue

    case_name       = case_dir.name                 # BST5, F1, …
    src_cfg_json    = case_dir / "config.json"
    src_cfg_yaml    = case_dir / "config.yaml"

    if not src_cfg_json.exists():
        print(f"[SKIP] Missing JSON: {src_cfg_json}")
        continue
    if not src_cfg_yaml.exists():
        print(f"[SKIP] Missing YAML: {src_cfg_yaml}")
        continue

    # ── Load & patch source JSON ──
    try:
        data = json.loads(src_cfg_json.read_text(encoding="utf-8"))
        _cfg_fname = data["logical_blueprint"]["config_filename"]
        if not isinstance(_cfg_fname, str):
            raise TypeError("config_filename is not a string")
    except Exception as e:
        print(f"[ERROR] {src_cfg_json}: {e}")
        continue

    # Replace with path to 'blueprint.json'
    data["logical_blueprint"]["config_filename"] = str(
        CONFIG_PREFIX / case_name / "blueprint.json"
    )

    # ── Prepare destination paths ──
    dst_dir         = DST_BASE / case_name
    dst_blueprint   = dst_dir / "blueprint.json"   # <- new name
    dst_yaml        = dst_dir / "config.yaml"
    dst_test_info   = dst_dir / "test_info.json"

    if not dst_dir.exists() and not DRY_RUN:
        dst_dir.mkdir(parents=True, exist_ok=True)

    # ── Write blueprint.json ──
    safe_action(
        "WRITE", dst_blueprint,
        lambda: dst_blueprint.write_text(json.dumps(data, indent=2), encoding="utf-8")
    )

    # ── Copy config.yaml ──
    safe_action(
        "COPY ", dst_yaml,
        lambda: shutil.copy2(src_cfg_yaml, dst_yaml)
    )

    # ── Copy matching info file → test_info.json ──
    info_src = info_index.get(case_name)
    if info_src and info_src.is_file():
        safe_action(
            "COPY ", dst_test_info,
            lambda: shutil.copy2(info_src, dst_test_info)
        )
    else:
        print(f"[MISS] No info file for {case_name} in {INFO_SRC_ROOT}")

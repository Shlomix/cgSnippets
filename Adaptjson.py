import json
import shutil
from pathlib import Path
from itertools import chain

# ──► EDIT THESE THREE PATHS ◄──
SRC_BASE      = Path("/absolute/path/to/cfg_debugs").resolve(strict=True)
DST_BASE      = Path("/root/blue/tubes/boost_sdk/boost_tests_experimental/tests_blueprint")
CONFIG_PREFIX = Path("/root/blue/tubes/boost_sdk/boost_tests_experimental")
INFO_SRC_ROOT = Path("/absolute/path/to/info_files").resolve(strict=True)   # NEW

# ──► SAFETY FLAGS ◄──
DRY_RUN   = False      # True = simulate only
OVERWRITE = False      # False = never replace existing dst files

# ────────────────────────────────────────────────────────────────────────────────
# 1. Pre-index all candidate info files once (fast & deterministic)
info_index = {}                     # { 'BST1': Path(.../BST1.json), 'F1': Path(.../F1.json) }
for p in INFO_SRC_ROOT.rglob("*.json"):
    stem = p.stem                                  # file name without extension
    # Only keep the first occurrence; warn on duplicates
    if stem not in info_index:
        info_index[stem] = p
    else:
        print(f"[WARN] Duplicate info file for {stem}: {p} (already have {info_index[stem]})")

# 2. Collect all case directories we need to process (BST* + F1/F2)
pattern_dirs = SRC_BASE.glob("BST*")
manual_dirs  = [SRC_BASE / "F1", SRC_BASE / "F2"]
all_dirs = sorted(chain(pattern_dirs, manual_dirs))

# 3. Main loop
for case_dir in all_dirs:
    if not case_dir.exists():
        print(f"[SKIP] Directory does not exist: {case_dir}")
        continue

    case_name = case_dir.name           # e.g. BST3 or F2
    cfg_json  = case_dir / "config.json"
    cfg_yaml  = case_dir / "config.yaml"

    # ── Validation ──
    if not cfg_json.exists():
        print(f"[SKIP] Missing JSON: {cfg_json}")
        continue
    if not cfg_yaml.exists():
        print(f"[SKIP] Missing YAML: {cfg_yaml}")
        continue

    # ── Read / patch config.json ──
    try:
        data = json.loads(cfg_json.read_text(encoding="utf-8"))
        cfg_field = data["logical_blueprint"]["config_filename"]
        if not isinstance(cfg_field, str):
            raise TypeError("config_filename is not a string")
    except Exception as e:
        print(f"[ERROR] {cfg_json}: {e}")
        continue

    new_cfg_filename = str(CONFIG_PREFIX / case_name / Path(cfg_field).name)
    data["logical_blueprint"]["config_filename"] = new_cfg_filename

    # ── Prepare destination paths ──
    dst_dir         = DST_BASE / case_name
    dst_json        = dst_dir / "config.json"
    dst_yaml        = dst_dir / "config.yaml"
    dst_test_info   = dst_dir / "test_info.json"

    # Create destination dir
    if not dst_dir.exists() and not DRY_RUN:
        dst_dir.mkdir(parents=True, exist_ok=True)

    # ── Helper to guard overwrites ──
    def safe_write(action: str, src: Path, dst: Path, writer):
        if dst.exists() and not OVERWRITE:
            print(f"[SKIP] {action} exists and OVERWRITE=False: {dst}")
            return
        print(f"[{action}] {dst}")
        if not DRY_RUN:
            writer()

    # ── Write adjusted config.json ──
    safe_write("WRITE", None, dst_json,
               lambda: dst_json.write_text(json.dumps(data, indent=2), encoding="utf-8"))

    # ── Copy config.yaml ──
    safe_write("COPY ", cfg_yaml, dst_yaml,
               lambda: shutil.copy2(cfg_yaml, dst_yaml))

    # ── Locate and copy test_info file ──
    info_src = info_index.get(case_name)
    if info_src and info_src.is_file():
        safe_write("COPY ", info_src, dst_test_info,
                   lambda: shutil.copy2(info_src, dst_test_info))
    else:
        print(f"[MISS] No info file found for {case_name} in {INFO_SRC_ROOT}")

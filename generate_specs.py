#!/usr/bin/env python3
"""Generate one JSON file per (tp, pp, remotes) combo."""
import json, itertools, pathlib, shutil

spec_dir = pathlib.Path("allocator_specs")
if spec_dir.exists(): shutil.rmtree(spec_dir)
spec_dir.mkdir()

tp_vals, pp_vals, rem_vals = (1,2), (1,2), (0,1,2)

def remote_cap(r): return None if r==0 else (2 if r==1 else 1)

for tp, pp, rem in itertools.product(tp_vals, pp_vals, rem_vals):
    spec = {
        "name": f"tp{tp}_pp{pp}_rem{rem}",
        "tp": tp,
        "pp": pp,
        "remotes": rem,
        "remote_max_tables": remote_cap(rem),
        "page_size": 16,
        "pages_per_table": 2,
        "warmed_up": True,
        "sequences": "auto:4x16",
        "decode_steps": 4
    }
    (spec_dir / f"{spec['name']}.json").write_text(json.dumps(spec, indent=2))

print(f"Generated {len(list(spec_dir.iterdir()))} spec files in {spec_dir}/")

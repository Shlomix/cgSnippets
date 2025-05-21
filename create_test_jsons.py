#!/usr/bin/env python3
"""
generate_specs.py
=================
Create 12 JSON spec files—one for every combination:

    tp ∈ {1, 2}
    pp ∈ {1, 2}
    remotes ∈ {0, 1, 2}

Each spec encodes:
  • page_size      = 16
  • pages_per_table= 2
  • warmed-up flag (handled explicitly in tests now)
  • remote_max_tables:  None | 2 | 1   (per earlier capacity rules)
  • sequences: shorthand  "auto:4x16"  (expanded by your runner)
  • decode_steps: 4
Files are written to ./allocator_specs/<name>.json
"""

import json, itertools, pathlib, shutil

# --------------------------------------------------------------------- #
spec_dir = pathlib.Path("allocator_specs")
if spec_dir.exists():
    shutil.rmtree(spec_dir)
spec_dir.mkdir()

def remote_cap(rem: int) -> int | None:
    """Rule: 0 → None, 1 → 2 tables, 2 → 1 table."""
    return None if rem == 0 else (2 if rem == 1 else 1)

count = 0
for tp, pp, rem in itertools.product((1, 2), (1, 2), (0, 1, 2)):
    spec = {
        "name": f"tp{tp}_pp{pp}_rem{rem}",
        "tp": tp,
        "pp": pp,
        "remotes": rem,
        "remote_max_tables": remote_cap(rem),
        "page_size": 16,
        "pages_per_table": 2,
        "sequences": "auto:4x16",   # your runner expands this
        "decode_steps": 4
    }
    (spec_dir / f"{spec['name']}.json").write_text(json.dumps(spec, indent=2))
    count += 1

print(f"✅  Generated {count} spec files in {spec_dir}/")

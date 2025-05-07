import json, os

OUT = "tests"
os.makedirs(OUT, exist_ok=True)

PAGE = 16
CAP  = 2   # pages per table  → 32 tokens

def write(name, workers, prompt, steps, expect):
    with open(f"{OUT}/{name}.json", "w") as f:
        json.dump(
            dict(
                name=name,
                workers=workers,
                prompt_len=prompt,
                page_size=PAGE,
                table_cap=CAP,
                decode_steps=steps,
                expected_workers=expect,
            ),
            f,
            indent=2,
        )

# local only
write("local_pref", {-1: 4, 0: 0}, 16, 0, [-1])
write("local_dec",  {-1: 4, 0: 0}, 16, 1, [-1])

# remote only
write("remote_pref", {-1: 0, 0: 4}, 16, 0, [0])
write("remote_dec",  {-1: 0, 0: 4}, 16, 1, [0])

# local + remote
write("both_pref", {-1: 1, 0: 4}, 16, 0, [-1, 0])
write("both_dec",  {-1: 1, 0: 4}, 16, 1, [-1, 0])

print("✓ JSON cases written into ./tests/")

#!/usr/bin/env python3
# send_prompt.py
# A tiny client for an OpenAI-compatible /v1/completions server.

import argparse, json, os, sys, time, string, random

DEFAULT_URL = "http://localhost:8000/v1/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct"

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def random_words(n: int, seed: int | None = None, min_len: int = 3, max_len: int = 10) -> str:
    """Generate n pseudo-words; good for large prompt testing."""
    rng = random.Random(seed)
    letters = string.ascii_lowercase
    words = []
    for i in range(n):
        L = rng.randint(min_len, max_len)
        words.append("".join(rng.choice(letters) for _ in range(L)))
    # light punctuation every ~20 words so it looks more natural
    for i in range(19, len(words), 20):
        words[i] += rng.choice([".", ",", "?", "!", ":"])
    return " ".join(words)

def random_chars(n: int, seed: int | None = None) -> str:
    """Generate n arbitrary ASCII chars (letters+space); reproducible with seed."""
    rng = random.Random(seed)
    alphabet = string.ascii_letters + " "
    return "".join(rng.choice(alphabet) for _ in range(n))

def post_json(url: str, payload: dict, timeout: int = 300) -> tuple[dict, float]:
    try:
        import requests  # type: ignore
        t0 = time.time()
        r = requests.post(url, headers={"Content-Type": "application/json"},
                          data=json.dumps(payload), timeout=timeout)
        dt = time.time() - t0
        r.raise_for_status()
        return r.json(), dt
    except ModuleNotFoundError:
        # stdlib fallback
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError
        req = Request(url, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                dt = time.time() - t0
                return json.loads(body), dt
        except HTTPError as e:
            sys.stderr.write(f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}\n")
            sys.exit(1)
        except URLError as e:
            sys.stderr.write(f"URL error: {e}\n")
            sys.exit(1)

def extract_text(resp: dict) -> str:
    # OpenAI /v1/completions shape
    try:
        return resp["choices"][0].get("text", "").rstrip()
    except Exception:
        return json.dumps(resp, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Simple client for /v1/completions")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"Endpoint URL (default: {DEFAULT_URL})")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")

    # prompt sources
    ap.add_argument("--prompt", default=None, help="Prompt string (optional)")
    ap.add_argument("--file", default=None, help="Read prompt from file (optional)")
    ap.add_argument("--random-words", type=int, default=None, metavar="N",
                    help="Generate a random prompt of N words")
    ap.add_argument("--random-chars", type=int, default=None, metavar="N",
                    help="Generate a random prompt of N characters")
    ap.add_argument("--seed", type=int, default=None, help="Seed for random prompt (optional)")
    ap.add_argument("--order", default="file,random,prompt",
                    help="Concatenation order of sources (comma separated subset of file,random,prompt). "
                         "Default: file,random,prompt")

    # generation knobs
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--stop", action="append", default=None,
                    help="Add a stop string (can be passed multiple times)")

    # execution
    ap.add_argument("--n", type=int, default=1, help="Number of requests to send (repeat)")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--verbose", action="store_true", help="Print payload and latency")
    ap.add_argument("--raw", action="store_true", help="Print raw JSON response instead of text")
    args = ap.parse_args()

    # Build the prompt parts
    sources = []
    order = [s.strip() for s in args.order.split(",") if s.strip() in {"file", "random", "prompt"}]

    if "file" in order and args.file:
        sources.append(("file", read_file(args.file)))

    rand_text = None
    if "random" in order:
        if args.random_words is not None and args.random_chars is not None:
            sys.stderr.write("Choose either --random-words or --random-chars, not both.\n")
            sys.exit(2)
        if args.random_words is not None:
            rand_text = random_words(args.random_words, seed=args.seed)
        elif args.random_chars is not None:
            rand_text = random_chars(args.random_chars, seed=args.seed)
        if rand_text:
            sources.append(("random", rand_text))

    if "prompt" in order and args.prompt:
        sources.append(("prompt", args.prompt))

    if not sources:
        sys.stderr.write("Provide at least one source: --prompt or --file or --random-words/--random-chars\n")
        sys.exit(2)

    # Concatenate with a single newline between sources, preserving the order chosen
    prompt = "\n".join(text for _, text in sources)

    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.stop:
        payload["stop"] = args.stop

    # Send requests
    for i in range(args.n):
        if args.verbose:
            src_summary = ", ".join(f"{name}({len(text)})" for name, text in sources)
            print(f"# Request {i+1}/{args.n} → {args.url}")
            print(f"# Sources: {src_summary}")
            print(f"# Payload: {json.dumps({k: payload[k] for k in payload if k!='prompt'}, ensure_ascii=False)}")
            print(f"# Prompt preview (first 240 chars): {prompt[:240].replace(os.linesep,' ')}"
                  f"{'…' if len(prompt) > 240 else ''}")

        resp, dt = post_json(args.url, payload, timeout=args.timeout)

        if args.verbose:
            print(f"# Latency: {dt*1000:.1f} ms")

        if args.raw:
            print(json.dumps(resp, ensure_ascii=False, indent=2))
        else:
            print(extract_text(resp))

if __name__ == "__main__":
    main()

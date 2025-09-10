#!/usr/bin/env python3
# send_prompt.py
import argparse, json, sys, os

DEFAULT_URL = "http://localhost:8000/v1/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct"

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def post_json(url: str, payload: dict) -> dict:
    try:
        import requests  # type: ignore
        r = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=300)
        r.raise_for_status()
        return r.json()
    except ModuleNotFoundError:
        # fallback to stdlib
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError
        req = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=300) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                return json.loads(body)
        except HTTPError as e:
            sys.stderr.write(f"HTTP error {e.code}: {e.read().decode('utf-8', errors='ignore')}\n")
            sys.exit(1)
        except URLError as e:
            sys.stderr.write(f"URL error: {e}\n")
            sys.exit(1)

def extract_text(resp: dict) -> str:
    # Try OpenAI-style "completions" shape first
    try:
        return resp["choices"][0].get("text", "").strip()
    except Exception:
        # Fallback: dump full JSON if unexpected
        return json.dumps(resp, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Simple client for /v1/completions")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"Endpoint URL (default: {DEFAULT_URL})")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--prompt", default=None, help="Prompt string (optional)")
    ap.add_argument("--file", default=None, help="Read prompt from file (optional)")
    ap.add_argument("--max_tokens", type=int, default=64, help="Max tokens to generate (default: 64)")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    ap.add_argument("--top_p", type=float, default=None, help="Top-p (optional)")
    ap.add_argument("--seed", type=int, default=None, help="Seed (optional)")
    args = ap.parse_args()

    parts = []
    if args.file:
        parts.append(read_file(args.file))
    if args.prompt:
        parts.append(args.prompt)
    if not parts:
        sys.stderr.write("Provide --prompt or --file\n")
        sys.exit(2)

    prompt = "\n".join(p.strip() for p in parts if p is not None)

    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.seed is not None:
        payload["seed"] = args.seed

    resp = post_json(args.url, payload)
    text = extract_text(resp)
    print(text)

if __name__ == "__main__":
    main()

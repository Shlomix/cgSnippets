#!/usr/bin/env python3
# inject_long_story_file.py
import os, argparse, time, json, csv, sys
import requests

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_payload(endpoint: str, model: str, prompt: str, max_new_tokens: int, temperature: float, stream: bool):
    if endpoint == "completions":
        return {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": stream,
        }
    elif endpoint == "chat":
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": stream,
        }
    elif endpoint == "generate":  # vLLM native fallback
        return {
            "prompt": prompt,
            "n": 1,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            # no stream support here
        }
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}")

def endpoint_path(endpoint: str) -> str:
    if endpoint == "completions":
        return "/v1/completions"
    if endpoint == "chat":
        return "/v1/chat/completions"
    if endpoint == "generate":
        return "/generate"
    raise ValueError(endpoint)

def extract_usage(endpoint: str, data) -> dict:
    """Return {'prompt_tokens': int|None, 'completion_tokens': int|None} when available."""
    usage = {"prompt_tokens": None, "completion_tokens": None}
    try:
        if endpoint in ("completions", "chat"):
            # OpenAI-compatible responses
            if isinstance(data, dict) and "usage" in data and isinstance(data["usage"], dict):
                usage["prompt_tokens"] = data["usage"].get("prompt_tokens")
                usage["completion_tokens"] = data["usage"].get("completion_tokens")
        elif endpoint == "generate":
            # Some vLLM /generate builds return token counts; many do not.
            if isinstance(data, dict):
                usage["prompt_tokens"] = data.get("prompt_tokens") or None
                usage["completion_tokens"] = data.get("generated_tokens") or None
    except Exception:
        pass
    return usage

def run_once(base_url: str, endpoint: str, headers: dict, payload: dict, stream: bool, timeout: int):
    url = base_url.rstrip("/") + endpoint_path(endpoint)

    result = {
        "http_status": None,
        "ttfb_s": None,          # only when stream=True (OpenAI endpoints)
        "latency_s": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": None,
    }

    try:
        t0 = time.perf_counter()
        if stream and endpoint in ("completions", "chat"):
            # Streaming: measure TTFB and total latency
            r = requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout)
            result["http_status"] = r.status_code
            r.raise_for_status()

            ttfb = None
            # Iterate lines until first data stanza and final DONE
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("data: "):
                    if raw == "data: [DONE]":
                        # end of stream
                        break
                    if ttfb is None:
                        ttfb = time.perf_counter() - t0
                        result["ttfb_s"] = ttfb
            # When stream ends, measure total
            result["latency_s"] = time.perf_counter() - t0

            # Optionally, get usage via a non-streaming follow-up? Not necessary.
            # Some servers also include "x-usage" headers, but we’ll skip that for portability.
        else:
            # Non-streaming (simpler and fine for prefill testing with max_tokens=1)
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            result["http_status"] = r.status_code
            r.raise_for_status()
            t1 = time.perf_counter()
            result["latency_s"] = t1 - t0
            # Try to parse usage
            try:
                data = r.json()
            except Exception:
                data = None
            usage = extract_usage(endpoint, data)
            result.update(usage)
    except Exception as e:
        result["error"] = str(e)

    return result

def main():
    ap = argparse.ArgumentParser(description="Inject a long story into vLLM server and time it.")
    ap.add_argument("--file", required=True, help="Path to the text file containing the story/prompt.")
    ap.add_argument("--base-url", default=os.environ.get("VLLM_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", "qwen2_5"))
    ap.add_argument("--endpoint", choices=["completions", "chat", "generate"], default="completions",
                    help="API endpoint to use. Prefer 'completions'.")
    ap.add_argument("--max-new-tokens", type=int, default=1, help="Small decode to stress prefill.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""), help="Optional Bearer token.")
    ap.add_argument("--repeat", type=int, default=1, help="Number of times to send the same story.")
    ap.add_argument("--pause-ms", type=int, default=0, help="Pause between runs (ms).")
    ap.add_argument("--stream", action="store_true", help="Measure TTFB using streaming (OpenAI endpoints only).")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--csv", default="", help="Optional path to write per-run metrics as CSV.")
    args = ap.parse_args()

    prompt = read_text_file(args.file)
    if not prompt.strip():
        print("ERROR: The file is empty.", file=sys.stderr)
        sys.exit(1)

    # Build headers and payload prototype
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    payload = build_payload(args.endpoint, args.model, prompt, args.max_new_tokens, args.temperature, args.stream)

    rows = []
    for i in range(args.repeat):
        res = run_once(args.base_url, args.endpoint, headers, payload, args.stream, args.timeout)
        # Simple derived metrics
        total_tok = None
        if isinstance(res.get("prompt_tokens"), int) and isinstance(res.get("completion_tokens"), int):
            total_tok = res["prompt_tokens"] + res["completion_tokens"]

        tok_per_sec = None
        prefill_tok_per_sec = None
        if res["latency_s"] and total_tok:
            tok_per_sec = total_tok / res["latency_s"]
        if res["latency_s"] and isinstance(res.get("prompt_tokens"), int):
            prefill_tok_per_sec = res["prompt_tokens"] / res["latency_s"]

        print(f"[run {i+1}/{args.repeat}] HTTP={res['http_status']} "
              f"latency={res['latency_s']:.3f}s "
              f"{'(TTFB='+format(res['ttfb_s'],'.3f')+'s)' if res['ttfb_s'] else ''} "
              f"prompt_tok={res['prompt_tokens']} "
              f"new_tok={res['completion_tokens']} "
              f"tok/s~{tok_per_sec:.1f if tok_per_sec else 'n/a'} "
              f"prefill_tok/s~{prefill_tok_per_sec:.1f if prefill_tok_per_sec else 'n/a'} "
              f"{'ERR='+res['error'] if res['error'] else ''}")

        row = {
            "run": i + 1,
            "http_status": res["http_status"],
            "latency_s": res["latency_s"],
            "ttfb_s": res["ttfb_s"],
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "tok_per_sec": tok_per_sec,
            "prefill_tok_per_sec": prefill_tok_per_sec,
            "error": res["error"],
        }
        rows.append(row)

        if args.pause_ms > 0 and i < args.repeat - 1:
            time.sleep(args.pause_ms / 1000.0)

    # Aggregate summary
    latencies = [r["latency_s"] for r in rows if isinstance(r["latency_s"], (int, float))]
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        print(f"\nAvg latency over {len(latencies)} runs: {avg_lat:.3f}s")
    if any(r.get("tok_per_sec") for r in rows):
        vals = [r["tok_per_sec"] for r in rows if r["tok_per_sec"]]
        if vals:
            print(f"Avg total tok/s (approx): {sum(vals)/len(vals):.1f}")
    if any(r.get("prefill_tok_per_sec") for r in rows):
        vals = [r["prefill_tok_per_sec"] for r in rows if r["prefill_tok_per_sec"]]
        if vals:
            print(f"Avg prefill tok/s (approx): {sum(vals)/len(vals):.1f}")

    # Optional CSV
    if args.csv:
        fieldnames = list(rows[0].keys()) if rows else ["run","http_status","latency_s","ttfb_s","prompt_tokens","completion_tokens","tok_per_sec","prefill_tok_per_sec","error"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Saved CSV: {os.path.abspath(args.csv)}")

if __name__ == "__main__":
    main()

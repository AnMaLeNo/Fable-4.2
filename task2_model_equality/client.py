"""Client for task 2. Every paid answer is appended to a jsonl log before anything else."""
import json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT = Path(__file__).parent
def _key():
    # credentials are read from the environment only (export SERVER_URL=... API_KEY=...); nothing is stored in the repository
    return os.environ["API_KEY"]
SERVER = os.environ["SERVER_URL"].rstrip("/")
BASE = f"{SERVER}/33-model-equality/v1"
S = requests.Session(); S.headers["X-API-Key"] = _key()
LOCK = threading.Lock()

def budget():
    r = S.get(f"{BASE}/models", timeout=60); r.raise_for_status()
    return {m["id"]: m["budget"] for m in r.json()["data"]}

def log_path(ep):
    return ROOT / ("ep_answers.jsonl" if ep.startswith("ep-") else "ref_answers.jsonl")

def load(path):
    p = ROOT / path
    if not p.exists(): return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]

def ask(ep, prompt, pid, rep=0, max_tokens=256, extra=None, messages=None, tag=""):
    """One request. Returns the logged record (or dict with 'error')."""
    payload = {"model": ep, "max_tokens": max_tokens,
               "messages": messages or [{"role": "user", "content": prompt}]}
    if extra: payload.update(extra)
    t0 = time.time()
    while time.time() - t0 < 3600:
        try:
            r = S.post(f"{BASE}/chat/completions", json=payload, timeout=300)
        except requests.RequestException as e:
            time.sleep(15); continue
        if r.status_code == 200:
            body = r.json()
            rec = {"endpoint": ep, "id": pid, "repeat": rep, "prompt": prompt, "max_tokens": max_tokens,
                   "response": body["choices"][0]["message"]["content"], "ts": time.time(), "tag": tag,
                   "extra": extra, "body": body}
            with LOCK:
                with log_path(ep).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
            return rec
        if r.status_code == 429 and "Retry-After" in r.headers:
            time.sleep(1.0); continue
        if r.status_code in (502, 503, 504):
            if ep.startswith("ep-"):   # a 5xx may still be charged to the budget: never resend a scored-endpoint request blindly
                with LOCK:
                    with (ROOT / "ep_errors.jsonl").open("a") as f:
                        f.write(json.dumps({"endpoint": ep, "id": pid, "repeat": rep, "status": r.status_code, "ts": time.time()}) + "\n")
                return {"error": r.status_code, "detail": r.text[:200], "endpoint": ep, "id": pid}
            time.sleep(10); continue
        return {"error": r.status_code, "detail": r.text[:500], "endpoint": ep, "id": pid}
    return {"error": "retries", "endpoint": ep, "id": pid}

def run(jobs, parallel=4, quiet=False):
    """jobs: list of dict(ep, prompt, pid, rep, max_tokens, tag). Skips those already logged."""
    done = set()
    for path in ("ep_answers.jsonl", "ref_answers.jsonl"):
        for r in load(path):
            done.add((r["endpoint"], r["id"], r["repeat"]))
    todo = [j for j in jobs if (j["ep"], j["pid"], j.get("rep", 0)) not in done]
    print(f"{len(jobs)} jobs, {len(todo)} to query", flush=True)
    res = []
    def one(j):
        out = ask(j["ep"], j["prompt"], j["pid"], j.get("rep", 0), j.get("max_tokens", 256), tag=j.get("tag", ""))
        if "error" in out: print("ERR", out, flush=True)
        return out
    with ThreadPoolExecutor(parallel) as pool:
        for i, o in enumerate(pool.map(one, todo)):
            res.append(o)
            if (i + 1) % 50 == 0: print(f"{i+1}/{len(todo)}", flush=True)
    return res

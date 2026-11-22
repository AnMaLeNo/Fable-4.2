"""python submit.py FILE.pt  -> prints full server reply, appends to submissions.log"""
import sys, json, time, requests
from pathlib import Path
import os
# credentials are read from the environment only (export SERVER_URL=... API_KEY=...); nothing is stored in the repository
env = {"SERVER_URL": os.environ["SERVER_URL"], "API_KEY": os.environ["API_KEY"]}
f = Path(sys.argv[1])
r = None
for attempt in range(6):
    try:
        with f.open("rb") as fh:
            r = requests.post(f"{env['SERVER_URL']}/submit/31-unlearning", headers={"X-API-Key": env["API_KEY"]},
                              files={"file": (f.name, fh)}, timeout=(180, 900))
        break
    except requests.RequestException as e:
        print("network error, retry in 30 s:", type(e).__name__, flush=True); time.sleep(30)
if r is None: raise SystemExit("submission failed: server unreachable")
try: body = r.json()
except Exception: body = r.text[:500]
line = json.dumps({"t": time.strftime("%H:%M:%S"), "file": str(f), "http": r.status_code, "body": body, "note": " ".join(sys.argv[2:])})
print(line); open("submissions.log", "a").write(line + "\n")

"""python submit.py file.csv  -> validates, submits, logs to subs/log.jsonl. Enforces the 30 min cooldown locally."""
import sys, csv, json, time, pathlib, requests, client
f = pathlib.Path(sys.argv[1])
rows = [r for r in csv.reader(f.open()) if r and r[0] != "endpoint"]
assert len(rows) == 33 and sorted(r[0] for r in rows) == [f"ep-{i:02d}" for i in range(1, 34)], "need 33 endpoints"
fam = {}
for e, fa, m in rows:
    assert "," not in fa + m and fa and m
    assert fam.setdefault(m, fa) == fa, f"model {m} in two families"
log = pathlib.Path("subs/log.jsonl")
last = [json.loads(l) for l in log.open()][-1]["t"] if log.exists() else 0
assert time.time() - last > 30 * 60 + 20, f"cooldown: wait {int(30*60+20-(time.time()-last))} s"
with log.open("a") as L:
    L.write(json.dumps({"t": time.time(), "file": str(f), "status": "sending"}) + "\n")
r = requests.post(f"{client.SERVER}/submit/33-model-equality", headers={"X-API-Key": client._key()},
                  files={"file": (f.name, f.open("rb"))}, timeout=(600, 600))
print(r.status_code, r.text[:500])
with log.open("a") as L:
    L.write(json.dumps({"t": time.time(), "file": str(f), "status": r.status_code, "body": r.text[:500]}) + "\n")

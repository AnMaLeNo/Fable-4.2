"""Phase 3 collection. python ep_q3.py FAMILY eps(comma) rep_from rep_to   (each rep = one response to qe1 and one to qe2)"""
import sys, json, time, client
sys.path.insert(0, "jr"); import pool
f, eps, r0, r1 = sys.argv[1], [f"ep-{int(i):02d}" for i in sys.argv[2].split(",")], int(sys.argv[3]), int(sys.argv[4])
QE = json.load(open(f"out/q3_{f}.json"))["QE"]
jobs = [dict(ep=e, prompt=pool.qprompt(QE[pid]), pid=pid, rep=r, max_tokens=256, tag="q3") for r in range(r0, r1) for pid in ("qe1", "qe2") for e in eps]
b = client.budget(); json.dump({k: v["used"] for k, v in b.items()}, open(f"counters_before_q3{f}_{int(time.time())}.json", "w"))
assert max(b[e]["used"] for e in eps) + 2 * (r1 - r0) <= 72, "would exceed 72 used"
client.run(jobs, parallel=4)
b = client.budget(); json.dump({k: v["used"] for k, v in b.items()}, open(f"counters_after_q3{f}_{int(time.time())}.json", "w"))
print("done")

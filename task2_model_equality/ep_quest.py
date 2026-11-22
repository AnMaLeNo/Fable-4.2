import sys, json, time, client, quest
which, r0, r1, par = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
p = quest.prompt(getattr(quest, which))
eps = [f"ep-{i:02d}" for i in range(1, 34)]
jobs = [dict(ep=e, prompt=p, pid=which.lower(), rep=r, max_tokens=256, tag="quest") for r in range(r0, r1) for e in eps]
b = client.budget(); json.dump({k: v["used"] for k, v in b.items()}, open(f"counters_before_{which}_{r0}_{int(time.time())}.json", "w"))
assert max(v["used"] for k, v in b.items() if k.startswith("ep")) + (r1 - r0) <= 45, "cap"
client.run(jobs, parallel=par)
b = client.budget(); json.dump({k: v["used"] for k, v in b.items()}, open(f"counters_after_{which}_{r0}_{int(time.time())}.json", "w"))
print("done")

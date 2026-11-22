import json, client, fam
rows = {"A": [], "B": [], "C": []}
for r in client.load("ep_answers.jsonl"):
    if r["tag"] == "free" or r["id"] == "b1-story":
        f = next(k for k, v in fam.FAM.items() if int(r["endpoint"][3:]) in v)
        rows[f].append({"ep": r["endpoint"], "id": r["id"], "rep": r["repeat"], "prompt": r["prompt"], "response": r["response"]})
for f, rs in rows.items():
    open(f"jr/free_rows_{f}.jsonl", "w").write("\n".join(json.dumps(x) for x in rs) + "\n"); print(f, len(rs))

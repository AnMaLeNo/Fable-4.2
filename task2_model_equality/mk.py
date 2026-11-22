"""python mk.py out.csv 'A:1,4,12|2|23' ...  -> groups separated by |, first group etc. Unlisted family members become singletons."""
import sys, csv, fam
out = sys.argv[1]; rows = []
for spec in sys.argv[2:]:
    f, g = spec.split(":"); seen = set()
    for k, grp in enumerate(g.split("|")):
        for i in map(int, grp.split(",")):
            assert i in fam.FAM[f] and i not in seen, (f, i); seen.add(i)
            rows.append((f"ep-{i:02d}", f, f"{f}_g{k}"))
    for i in fam.FAM[f]:
        if i not in seen: rows.append((f"ep-{i:02d}", f, f"{f}_s{i}"))
assert len(rows) == 33
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["endpoint", "family", "model"]); w.writerows(sorted(rows))
print(out, "written")

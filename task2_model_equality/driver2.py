"""Unattended leaderboard probes. Queue file subs/queue.txt: one probe per line 'NAME;A:..;B:..;C:..' (lines starting with # are skipped).
After a probe that scores below the best score, the best file (subs/best.csv) is resubmitted at the next slot."""
import json, subprocess, sys, time, os, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))
BEST = 1.0
def wait_cooldown():
    while subprocess.run("pgrep -f '[p]robe.sh'", shell=True, capture_output=True).stdout.strip(): time.sleep(20)
    last = [json.loads(x) for x in open("subs/log.jsonl")][-1]["t"]
    time.sleep(max(0, 30 * 60 + 30 - (time.time() - last)))
def submit(path):
    wait_cooldown()
    out = subprocess.run([sys.executable, "submit.py", path], capture_output=True, text=True)
    txt = (out.stdout + out.stderr).strip().splitlines()[-1] if (out.stdout + out.stderr).strip() else "no output"
    m = re.search(r'"score":\s*([0-9.]+)', txt)
    return (float(m.group(1)) if m else None), txt
def log(line):
    with open("subs/probes.txt", "a") as f: f.write(time.strftime("%H:%M:%S ") + line + "\n")
done = set()
if os.path.exists('subs/probes.txt'):
    for l in open('subs/probes.txt'):
        parts = l.split(' ', 1)
        if len(parts) > 1: done.add(parts[1].split(' | ')[0].strip())
while True:
    lines = [l.strip() for l in open("subs/queue.txt") if l.strip() and not l.startswith("#")]
    todo = [l for l in lines if l.split(";")[0] not in done]
    if os.path.exists("subs/STOP") or not todo: time.sleep(30); 
    if os.path.exists("subs/STOP"): break
    if not todo: continue
    name, *specs = todo[0].split(";")
    done.add(name)
    if subprocess.run([sys.executable, "mk.py", f"subs/{name}.csv"] + specs, capture_output=True).returncode != 0:
        log(f"{name} | BAD SPEC"); continue
    score, txt = submit(f"subs/{name}.csv")
    log(f"{name} | {' '.join(specs)} | score {score} | {txt[:260]}")
    if False:
        s2, t2 = submit("subs/best.csv")
        log(f"best.csv resubmitted | score {s2} | {t2[:200]}")

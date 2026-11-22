"""Phase 3: offline questionnaire design + power simulation with the exact base replica.
python q3.py MODEL TAG [n_quest n_design n_verify n_power]"""
import os, sys, json, time, pickle, collections, re
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantsim, pool, items
name, tag = sys.argv[1], sys.argv[2]
NQ, ND, NV, NP = [int(x) for x in sys.argv[3:7]] if len(sys.argv) > 6 else (10, 8, 16, 100)
dev = "cuda" if torch.cuda.is_available() else "cpu"; TAU = 0.7; BS = int(os.environ.get("BS", "25"))
tok = AutoTokenizer.from_pretrained(name)
def load(variant, eps=0.0):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    if variant == "sfp8": quantsim.apply(m, quantsim.scaled_fp8, "linear")
    if variant == "lowrank":
        g = torch.Generator().manual_seed(1234)
        for n_, p in m.named_parameters():
            if p.dim() == 2 and (n_.endswith("q_proj.weight") or n_.endswith("v_proj.weight")):
                B = torch.randn(p.shape[0], 8, generator=g); A = torch.randn(8, p.shape[1], generator=g); D = B @ A
                p.data += eps * p.data.norm() * D / D.norm()
    return m.to(dev)
def chat(prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
@torch.no_grad()
def gen(model, prompt, n, max_new=256):
    a = tok(chat(prompt), add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev); out = []
    while len(out) < n:
        b = min(BS, n - len(out))
        o = model.generate(a.repeat(b, 1), attention_mask=torch.ones(b, a.shape[1], dtype=torch.long, device=dev), do_sample=True, temperature=TAU, top_k=0, top_p=1.0,
                           repetition_penalty=1.0, max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
        for row in o[:, a.shape[1]:]:
            out.append(tok.decode(row, skip_special_tokens=True))
    return out
@torch.no_grad()
def score(model, prompt, text, ep, pid, rep):
    a = tok(chat(prompt), add_special_tokens=False)["input_ids"]; b = tok(text, add_special_tokens=False)["input_ids"][:300]
    if not b: return None
    lg = model(torch.tensor([a + b], device=dev)).logits[0, len(a) - 1: len(a) + len(b) - 1].float()
    top = torch.topk(lg, 12, -1); ar = torch.arange(len(b), device=dev); tgt = torch.tensor(b, device=dev)
    return {"ep": ep, "id": pid, "rep": rep, "tok": np.array(b, np.int32), "top_ids": top.indices.cpu().numpy().astype(np.int32),
            "top_lg": top.values.cpu().numpy().astype(np.float32), "lse": {TAU: torch.logsumexp(lg.double() / TAU, -1).cpu().numpy()},
            "lp": (torch.log_softmax(lg.double() / TAU, -1)[ar, tgt]).cpu().numpy()}
def item_stats(dump, n_items):
    """per (pid, item): mean Gini of the first-answer-token distribution, and how often the item was answered"""
    G = collections.defaultdict(list)
    for r in dump:
        for t, item in items.answer_positions(tok, r["tok"]):
            p = np.exp(r["top_lg"][t].astype(np.float64) / TAU - r["lse"][TAU][t]); G[(r["id"], item)].append(1 - float((p ** 2).sum()))
    return G
t0 = time.time(); base = load("plain")
# ---------- design
P = pool.pool(); quests = [P[i * 40:(i + 1) * 40] for i in range(NQ)]
dump = []
for k, qs in enumerate(quests):
    pr = pool.qprompt(qs)
    for j, tx in enumerate(gen(base, pr, ND)):
        s = score(base, pr, tx, "design", f"d{k}", j)
        if s: dump.append(s)
G = item_stats(dump, 40); cand = []
for k, qs in enumerate(quests):
    for i, q in enumerate(qs):
        g = G.get((f"d{k}", i + 1), [])
        cand.append((np.mean(g) if len(g) >= max(2, ND // 2) else -1.0, len(g), q))
cand.sort(reverse=True)
sel = [c[2] for c in cand[:80]]
QE = {"qe1": sel[0::2], "qe2": sel[1::2]}
print(tag, "design done", round(time.time() - t0), "s; candidate items", len(cand), "top Gini", [round(c[0], 2) for c in cand[:80:10]], flush=True)
# ---------- verify + power
VAR = [("base", "plain", 0.0), ("sfp8", "sfp8", 0.0), ("lr1", "lowrank", float(os.environ.get("E1", "0.01"))), ("lr2", "lowrank", float(os.environ.get("E2", "0.03")))]
texts = {}; scored = []; klinfo = {}
for vname, vkind, eps in VAR:
    m = base if vname == "base" else load(vkind, eps)
    for pid, qs in QE.items():
        pr = pool.qprompt(qs); tx = gen(m, pr, NP); texts[(vname, pid)] = tx
        if vname != "base":   # KL per token estimate: variant text scored under variant minus under base
            d = []
            for t in tx[:20]:
                sv = score(m, pr, t, vname, pid, 0); sb = score(base, pr, t, vname, pid, 0)
                if sv and sb: d.append(float((sv["lp"] - sb["lp"]).sum()) / len(sv["lp"]))
            klinfo[(vname, pid)] = float(np.mean(d))
    if vname != "base": del m; torch.cuda.empty_cache()
    print(tag, "generated", vname, round(time.time() - t0), "s", flush=True)
for (vname, pid), tx in texts.items():
    for j, t in enumerate(tx):
        s = score(base, QE[pid] and pool.qprompt(QE[pid]), t, f"{vname}#{j // 10}", pid, j)   # replicate = 10 responses per prompt
        if s: scored.append(s)
G2 = item_stats([r for r in scored if r["ep"].startswith("base")], 40)
print(tag, "final questionnaires: mean Gini", {pid: round(float(np.mean([np.mean(G2.get((pid, i + 1), [0])) for i in range(40)])), 3) for pid in QE},
      "answered items per response", round(float(np.mean([len(items.answer_positions(tok, r['tok'])) for r in scored if r['ep'].startswith('base')])), 1), flush=True)
O, E, N = items.build(scored, tok, TAU, pids=("qe1", "qe2"))
res = collections.defaultdict(list)
for ep in sorted({r["ep"] for r in scored}):
    chi, df, _ = items.gof(O, E, N, ep); res[ep.split("#")[0]].append((chi - df) / np.sqrt(2 * max(df, 1)))
thr = float(np.max(res["base"])) if res["base"] else 3.0
print(tag, "POWER with 20 responses (10 per prompt): z of item-level chi2 under the base replica; threshold = max over base replicates =", round(thr, 2))
for v in ("base", "sfp8", "lr1", "lr2"):
    z = np.array(res[v]); kl = np.mean([klinfo.get((v, p), 0) for p in QE]) if v != "base" else 0.0
    print(f"   {v}: KL/token {kl:.4f}  z mean {z.mean():6.2f} sd {z.std():5.2f} min {z.min():6.2f} max {z.max():6.2f}  rejected (z>{max(thr,3):.1f}): {int((z > max(thr, 3)).sum())}/{len(z)}", flush=True)
json.dump({"QE": QE, "cand": [(float(a), int(b), c) for a, b, c in cand], "z": {k: [float(x) for x in v] for k, v in res.items()}}, open(f"out/q3_{tag}.json", "w"))
print(tag, "done", round(time.time() - t0), "s", flush=True)

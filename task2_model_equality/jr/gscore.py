"""Self-contained scorer for the GPU node. python gscore.py MODEL ROWS.jsonl OUT_PREFIX variant1,variant2,... [dtype]
ROWS: one json per line {"ep","id","rep","prompt","response"}. Output: OUT_PREFIX_<variant>.pkl = {ep: [score dicts]}."""
import os, sys, json, pickle, collections, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantsim
TAUS = np.array([0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25])
name, rows_file, prefix, variants = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[sys.argv[5] if len(sys.argv) > 5 else "f32"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
VAR = {"plain": None,
       "fp8_lin": lambda m: quantsim.apply(m, quantsim.fp8_e4m3, "linear"),
       "fp8_all": lambda m: quantsim.apply(m, quantsim.fp8_e4m3, "all"),
       "fp8_mat": lambda m: quantsim.apply(m, quantsim.fp8_e4m3, "matrices"),
       "sfp8_lin": lambda m: quantsim.apply(m, quantsim.scaled_fp8, "linear"),
       "sfp8_mat": lambda m: quantsim.apply(m, quantsim.scaled_fp8, "matrices"),
       "sfp8r_lin": lambda m: quantsim.apply(m, quantsim.scaled_fp8_rows, "linear"),
       "w8a8": lambda m: (quantsim.apply(m, quantsim.scaled_fp8, "linear"), quantsim.add_act_fp8(m)),
       "w8a8r": lambda m: (quantsim.apply(m, quantsim.scaled_fp8_rows, "linear"), quantsim.add_act_fp8(m)),
       "a8": lambda m: quantsim.add_act_fp8(m),
       "e5m2_lin": lambda m: quantsim.apply(m, quantsim.fp8_e5m2, "linear"),
       "int8_lin": lambda m: quantsim.apply(m, lambda w: quantsim.int_sym(w, 8), "linear"),
       "int8t_lin": lambda m: quantsim.apply(m, lambda w: quantsim.int_sym(w, 8, per_channel=False), "linear"),
       "int4_lin": lambda m: quantsim.apply(m, lambda w: quantsim.int_sym(w, 4), "linear"),
       "f16": lambda m: m.half().float(), "bf16": lambda m: m.to(torch.bfloat16).float()}
rows = [json.loads(l) for l in open(rows_file) if l.strip()]
tok = AutoTokenizer.from_pretrained(name)
def load(variant):
    if variant.startswith("bnb"):
        from transformers import BitsAndBytesConfig
        cfg = {"bnb8": BitsAndBytesConfig(load_in_8bit=True),
               "bnb4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
               "bnb4nf4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16),
               "bnb4nf4d": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)}[variant]
        return AutoModelForCausalLM.from_pretrained(name, quantization_config=cfg, device_map={"": 0}).eval()
    m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).eval()
    if VAR[variant]: VAR[variant](m)
    return m.to(dev)
@torch.no_grad()
def score(model, prompt, response, max_resp=300):
    p = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    a = tok(p, add_special_tokens=False)["input_ids"]; b = tok(response, add_special_tokens=False)["input_ids"][:max_resp]
    if not b: return None
    x = torch.tensor([a + b], device=dev)
    lg = model(x).logits[0, len(a) - 1: len(a) + len(b) - 1].float(); tgt = torch.tensor(b, device=dev); ar = torch.arange(len(b), device=dev)
    lp, e, v = [], [], []
    for tau in TAUS:
        l = torch.log_softmax(lg / float(tau), -1); pr = l.exp(); ee = (pr * l).sum(-1)
        lp.append(l[ar, tgt].cpu().numpy()); e.append(ee.cpu().numpy()); v.append(((pr * l * l).sum(-1) - ee * ee).cpu().numpy())
    rank = (lg > lg[ar, tgt][:, None]).sum(-1).cpu().numpy()
    if DUMP is not None:
        top = torch.topk(lg, 12, -1)
        DUMP.append({"ep": CUR[0], "id": CUR[1], "rep": CUR[2], "tok": np.array(b, np.int32), "top_ids": top.indices.cpu().numpy().astype(np.int32),
                     "top_lg": top.values.cpu().numpy().astype(np.float32), "obs_lg": lg[ar, tgt].cpu().numpy().astype(np.float32),
                     "lse": {t: torch.logsumexp(lg.double() / t, -1).cpu().numpy() for t in (0.6, 0.7, 0.8, 0.9, 1.0)}})
    if RES is not None:
        ep = CUR[0]
        if ep not in RES: RES[ep] = {"O": torch.zeros(lg.shape[1], dtype=torch.float64, device=dev), "E": {t: torch.zeros(lg.shape[1], dtype=torch.float64, device=dev) for t in RT}, "V": {t: torch.zeros(lg.shape[1], dtype=torch.float64, device=dev) for t in RT}}
        RES[ep]["O"].index_add_(0, tgt, torch.ones(len(b), dtype=torch.float64, device=dev))
        for t in RT:
            pr = torch.softmax(lg.double() / t, -1); RES[ep]["E"][t] += pr.sum(0); RES[ep]["V"][t] += (pr * (1 - pr)).sum(0)
    return {"lp": np.array(lp, np.float32), "e": np.array(e, np.float32), "v": np.array(v, np.float32), "rank": rank, "n": len(b)}
RT = [0.6, 0.7, 0.8, 1.0]; RES = None; CUR = [None, None, None]; DUMP = None
SKIP = set(open("skip.txt").read().split()) if os.path.exists("skip.txt") else set()
for variant in variants:
    if variant in SKIP: continue
    t0 = time.time()
    RES = {} if variant in os.environ.get("RESVAR", "plain").split(",") else None
    DUMP = [] if variant in os.environ.get("DUMPVAR", "").split(",") else None
    try:
        model = load(variant)
    except Exception as ex:
        print("FAILED to build", variant, type(ex).__name__, str(ex)[:300], flush=True); continue
    by = collections.defaultdict(list)
    for r in rows:
        CUR[0] = r["ep"]; CUR[1] = r["id"]; CUR[2] = r["rep"]
        s = score(model, r["prompt"], r["response"])
        if s: s["id"] = r["id"]; s["rep"] = r["rep"]; by[r["ep"]].append(s)
    pickle.dump({"taus": TAUS, "by": dict(by)}, open(f"{prefix}_{variant}.pkl", "wb"))
    if DUMP is not None:
        pickle.dump(DUMP, open(f"{prefix}_dump_{variant}.pkl", "wb")); DUMP = None
    if RES is not None:
        out = {}
        for ep, d in RES.items():
            keep = (d["E"][1.0] > 0.02) | (d["O"] > 0)
            idx = keep.nonzero()[:, 0]
            out[ep] = {"idx": idx.cpu().numpy(), "O": d["O"][idx].cpu().numpy(), "E": {t: d["E"][t][idx].cpu().numpy() for t in RT}, "V": {t: d["V"][t][idx].cpu().numpy() for t in RT}}
        pickle.dump(out, open(f"{prefix}_residuals.pkl" if variant == "plain" else f"{prefix}_residuals_{variant}.pkl", "wb")); RES = None
    print("done", variant, len(rows), "rows", round(time.time() - t0), "s", flush=True)
    del model; torch.cuda.empty_cache()

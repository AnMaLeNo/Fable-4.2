# Task 2 — Model equality testing (LLMs behind APIs)

33 endpoints `ep-01` … `ep-33`. For each one, give its **family** (same base model) and its exact **model** (same
checkpoint, same quantisation, same LoRA adapter). Decoding watermarks, system prompts and decoding strategies do *not*
change the identity. Score = 0.1 × MCC(family) + 0.9 × MCC(model). Hard budget: 100 answers per endpoint for the whole
team, 256 generated tokens per answer; nine practice endpoints `ref-01` … `ref-09` with known labels and no budget limit.

**Public score: 1.00** (reached 21 Nov, 22:31; every later configuration listed below also scores 1.00 on the public
pairs). The configuration we consider the most probable is `subs/k01_H7.csv`.

## Idea

Distances between answer distributions are not enough: on the practice endpoints a *quantised* model is closer to the
base than a *persona* or a *high-temperature* endpoint, although the last two are the same model. So instead of
clustering, we identify **what each endpoint is**:

1. **Families** from the answers to two 40-question one-word questionnaires (three clean blocks of 11).
2. **The exact base checkpoint of each family**, by scoring the collected answers under local candidate models: the
   right checkpoint has a much higher likelihood per token. Once the checkpoint and the default temperature are known,
   we hold a **local replica** that gives the exact probability of every token the endpoint produced.
3. **Exact statistical tests against the replica**, which need no extra budget: temperature fit, item-by-item
   goodness of fit, likelihood ratio against a quantised replica (fp8), direction of deviation compared with a known
   modified endpoint, and a context-keyed watermark signature.
4. **Positive identification rule**: merge an endpoint with the base only when a disguise that preserves identity is
   positively identified (system prompt, persona, watermark, temperature); otherwise keep it as its own model.
5. For the few endpoints the tests cannot separate, **one-change-at-a-time submissions** read as hypothesis tests.

## Pipeline, step by step

| # | Step | File → function / entry point |
|---|---|---|
| 1 | API client. Every paid answer is written to disk before anything else, never asked twice (cache by endpoint / prompt / repetition), at most 4 parallel requests, and a scored-endpoint request is never resent after a 5xx | `client.py` → `ask`, `run`, `budget` |
| 2 | Questionnaires QA / QB: 40 one-word questions in a single prompt, numbered answers; collection on the 33 endpoints with budget counters saved before and after | `quest.py` → `prompt`, `parse`; `ep_quest.py` |
| 3 | Families: mean Hamming distance between answer vectors; response-level permutation tests (MMD with a Hamming kernel); temperature fit against the pooled family distribution | `qanalyze.py`; `fam.py` → `freq`, `echo_rate` (`FAM` holds the three families); `perm.py` → `test`; `tfit.py` → `run` |
| 4 | Checkpoint identification: log-likelihood per token of the collected answers under local candidate models, on a grid of temperatures | `lscore.py` → `Scorer`, `summarize`; `lref.py` (practice endpoints), `lep.py` (scored endpoints) |
| 5 | Quantised replicas: weights of the linear layers rounded to float8 E4M3 with a per-tensor scale (the variant validated on the practice fp8 endpoint), plus the variants that were rejected | `quantsim.py` → `scaled_fp8`, `fp8_e4m3`, `int_sym`, `apply` |
| 6 | GPU scoring of all answers under base and replica weights, per-token statistics and probability dumps; tables of fitted temperature, goodness of fit and likelihood ratios | `jr/gscore.py` → `load`, `score`; `gtab.py`; `resid.py` → `zvec`, `best_t` |
| 7 | Exact item-level goodness of fit: for each questionnaire item, the observed first answer token against the replica's probability in the exact context of each answer (χ²/dof ≈ 1 for base weights at the tested temperature) | `items.py` → `answer_positions`, `build`, `gof` |
| 8 | Exact directional test: does a candidate deviate from the base replica in the same direction as a known modified endpoint? | `dirtest.py` |
| 9 | Context-keyed watermark signature: two answers of one endpoint that reach the same 4-token context pick the same next token more often than the replica predicts | `wmrep.py` → `run` |
| 10 | Free-text second pass on the ambiguous endpoints only (about 0.7 nat/token against about 0.15 for the questionnaires) | `ep_free.py`, `export_free.py` |
| 11 | Questionnaire designed offline with the replica: 408 candidate questions, keep the 80 whose first answer token is the most uncertain, then simulate the power of the tests | `jr/pool.py` → `pool`, `qprompt`; `jr/q3.py`; `ep_q3.py`, `export_q3.py` |
| 12 | Submission: build the CSV from a group specification, validate it, respect the 30-minute cooldown; queue of full configurations | `mk.py`; `submit.py`; `driver2.py` with `subs/queue.txt` |

`jr/` holds the scripts that ran on the GPU node (`jr/items.py` and `jr/quantsim.py` are the copies used there).

## Reproduce

```bash
export SERVER_URL=<hackathon server> API_KEY=<team token>      # read from the environment, never stored

python ep_quest.py QA 0 12 4 ; python ep_quest.py QB 0 11 4    # WARNING: spends the scored-endpoint budget
python fam.py A B C                                            # family structure
python lep.py <Qwen2.5-3B-Instruct> A 1,2,4,12,14,17,18,23,24,25,33 100 plain     # likelihood under a candidate base
python jr/gscore.py <base model> jr/ep_rows_C.jsonl out/epC plain,sfp8_lin        # GPU: base and fp8 replica
python gtab.py out/epC sfp8_lin                                 # fitted temperature, fit, likelihood ratios
python items.py out/epC_dump_plain.pkl <base model> 0.7         # item-level exact test
python dirtest.py out/epC_dump_plain.pkl <base model> ep-05     # directional test
python wmrep.py 4 out/epC_dump_plain.pkl                        # watermark signature
python mk.py subs/k01_H7.csv 'A:1,2,4,12,14,23,33' 'B:6,7,8,20,21,30,32' 'C:10,13,16,19,27,29,31'
python submit.py subs/k01_H7.csv
```

Not included (data): the collected answers `ep_answers.jsonl` / `ref_answers.jsonl` (one JSON record per paid answer:
endpoint, prompt id, repetition, prompt, answer text, timestamp, raw API body), the per-family exports `jr/*_rows_*.jsonl`
and the replica score pickles `out/*.pkl`. The scored-endpoint budget cannot be re-spent, so an exact re-run needs
those files.

## Results

Base checkpoints identified by likelihood per token (best candidate against the runner-up):

| Family | Endpoints | Base model | Log-likelihood / token |
|---|---|---|---|
| A | 01, 02, 04, 12, 14, 17, 18, 23, 24, 25, 33 | Qwen2.5-3B-Instruct | −0.33 (1.5B: −0.62) |
| B | 03, 06, 07, 08, 20, 21, 22, 26, 28, 30, 32 | Qwen2.5-1.5B-Instruct | −0.25 (0.5B: −0.58; 3B: −0.67) |
| C | 05, 09, 10, 11, 13, 15, 16, 19, 27, 29, 31 | Llama-3.2-1B-Instruct | −0.35 (3B: −0.71) |
| practice | ref-01 … ref-09 | SmolLM2-360M-Instruct | −0.83 (135M: −1.26) |

Default temperature 0.7, no top-k. Validation on the practice endpoints: only the fp8 endpoint prefers our fp8 replica
(likelihood ratio base − replica: −15.6, against +33.2 for the plain endpoint); the goodness of fit at temperature 0.7
rejects LoRA 0.7 and 1.5 (z = −8.2 and −21.0) but not LoRA 0.3 (−3.6).

Structure retained (endpoints not listed are singletons, i.e. their own model):

| Family | "Base" group (same model) | How each member was identified | Kept as distinct models in our answer |
|---|---|---|---|
| A | 02, 04, 14, 23, 33 (+ 01, 12: see below) | 02 system prompt, 04 watermark, 14 temperature ≈ 0.8, 23 persona, 33 temperature 1.0 | 17, 18, 24, 25 (25 refuted as base by a submission test; 24 slightly off the base fit) |
| B | 06, 07, 08, 20, 30, 32 (+ 21: see below) | 06 watermark, 07 temperature ≈ 0.9, 08 system prompt, 20 persona, 30 temperature 1.0, 32 by submission test | 22 (fp8, positive evidence), 28 (rejected at every temperature), 03, 26 (refuted by a submission test) |
| C | 10, 13, 16, 27, 29, 31 (+ 19: see below) | 10 watermark (signature z = 24.7), 13 base, 16 persona, 27 temperature ≈ 0.9, 29 system prompt, 31 temperature 1.0 | 05, 11, 09 (chain of modified models), 15 (fp8 likely) |

Public score by submission:

| Time | File | Public score | What changed |
|---|---|---|---|
| 18:20 | sub01 | 0.10 | three families of 11, almost everything merged inside a family |
| 18:47 | sub02 | 0.40 | persona and low-entropy (watermark) endpoints merged with the base |
| 19:18 | sub03 | 0.75 | rule "merge only positively identified disguises" |
| 21:30 | sub04 | 0.85 | local replicas: temperature endpoints merged (14, 07, 27, 30), 28 removed from the base group |
| 22:31 | `subs/p01_B32.csv` | **1.00** | ep-32 added to the base group of B |
| 01:34 | `subs/k01_H7.csv` | 1.00 | most probable full configuration: 7 "base" endpoints per family (A + 01 + 12, B + 21, C + 19) |

Budget actually used on the scored endpoints (server counters, out of 100): 71 on ep-01, 70 on ep-12, 68 on ep-19,
67 on ep-13, 65 on ep-09; 56–59 on 17, 18, 24, 25, 33; 47–48 on 03, 05, 11, 21, 22, 26, 30, 32; 35–41 on the others.
This includes 7 to 12 short answers per endpoint collected separately by a team member in the first hours.

## What did not work, and what is not resolved

- The public score takes very few values (0.10 / 0.40 / 0.75 / 0.80 / 0.85 / 1.00): few pairs are public. Since 22:31
  every plausible configuration scores 1.00 publicly, so the public score can no longer confirm a correct merge; it can
  only refute a wrong one (ep-25 in the base group of A: 1.00 → 0.80; ep-26 in place of ep-21 in B, or next to it:
  1.00 → 0.80). Removing 14, 27 or 07 from their base group leaves 1.00: these pairs are not public either.
- **Unresolved**: whether 01 and 12 (A), 21 (B) and 19 (C) belong to the base group. With 22 answers each to the
  offline-designed questionnaire, 01 and 12 stay indistinguishable from the base replica (χ²/dof 0.82 and 0.72).
  For 19 the indicators are intermediate between the base and a weakly modified model; the "odd/even" anomaly that
  first pointed at it (6 "odd" out of 11) did not reproduce (0 out of 10). The final ranking keeps our best
  submission on the hidden pairs among all the configurations we submitted (`subs/queue.txt`), so we submitted the
  plausible readings in decreasing order of probability.
- Simulated power of the global tests with 20 answers: perturbations below about 0.03 nat/token (an fp8 replica is at
  0.008–0.024) are not detected; only directed tests (likelihood ratio against a replica, direction of a known
  modified endpoint) see them. No fp8 endpoint was found in family A.
- Abandoned: "line through the origin" test against the temperature endpoint (rejects everything, z = 3 to 6); binary
  choices (deterministic in family A); response times (identical on all practice endpoints, 24.2–24.9 s); a default
  KGW detector on the practice watermark endpoint (z = 2.1); naive float8 cast without scale, int8 / int4, bitsandbytes
  8-bit / 4-bit / nf4 replicas (no endpoint prefers any of them).
- Limitation of the method: it relies on the public availability of the base checkpoints and on the practice endpoints
  to learn what each disguise looks like.

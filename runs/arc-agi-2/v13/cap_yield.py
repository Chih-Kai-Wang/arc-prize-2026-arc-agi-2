"""v4 planning (zero GPU): verified-program yield per LLM window vs reasoning cap, from E3 + v2 dev-run job rows.
Censored fits (lognormal, Weibull) of natural completion length (stops observed 42.9k-59.4k; 66 jobs censored at 64k),
per-job decode speed ~constant (E3: 18.6 tok/s at 42.9k, 18.1 at 59.4k, 17.8 at 64k; conc 13, 226 tok/s aggregate),
KV: E3 server 9.13x concurrency at 131,072 tokens -> ~1.197M token-slots; 13 x 64k peak 0.695 (observed 0.695).
  uv run --quiet --no-project --with scipy --with numpy python -I runs/arc-agi-2/v13/cap_yield.py ROOT"""
import json, math, sys
import numpy as np
from scipy import stats, optimize, integrate

root = sys.argv[1]
obs, cens, trunc_ver = [], 0, 0
for name in ("v13-e3-seeds-v1", "v13-progsynth-v2"):
    for l in open(f"{root}/runs/arc-agi-2/v6/output-arc-2026-{name}/v13-programs.jsonl"):
        r = json.loads(l) if l.strip() else None
        if not r or r["state"] != "done":
            continue
        if r["finish_reason"] == "stop":
            obs.append(int(r["completion_tokens"]))
        elif r["finish_reason"] == "length":
            cens += 1
            trunc_ver += bool(r.get("verified"))
obs = np.array(obs, float)
v_trunc = trunc_ver / cens                     # reasoning-draft verification of truncated jobs (v2 behaviour, no forcing)
C = 64000.0


def fit_logn(o, c):
    f = lambda th: -(stats.norm.logpdf(np.log(o), th[0], math.exp(th[1])).sum() + c * stats.norm.logsf(math.log(C), th[0], math.exp(th[1])))
    x = optimize.minimize(f, [11.4, math.log(0.33)], method="Nelder-Mead").x
    return stats.lognorm(s=math.exp(x[1]), scale=math.exp(x[0]))


def fit_weib(o, c):
    f = lambda th: -(stats.weibull_min.logpdf(o, math.exp(th[0]), scale=math.exp(th[1])).sum()
                     + c * stats.weibull_min.logsf(C, math.exp(th[0]), scale=math.exp(th[1])))
    x = optimize.minimize(f, [math.log(5), math.log(88000)], method="Nelder-Mead").x
    return stats.weibull_min(math.exp(x[0]), scale=math.exp(x[1]))


def boot_low(q=10, B=300):  # pessimistic lognormal: 10th percentile of P(stop<=96k) over bootstrap fits
    rng = np.random.default_rng(0)
    allv = np.concatenate([obs, np.full(cens, np.inf)])
    fits = []
    for _ in range(B):
        s = rng.choice(allv, len(allv))
        o = s[np.isfinite(s)]
        if len(o) >= 3:
            d = fit_logn(o, int((~np.isfinite(s)).sum()))
            fits.append((d.cdf(96000), d))
    fits.sort(key=lambda x: x[0])
    return fits[int(len(fits) * q / 100)][1]


SPEED = 18.0           # tok/s per job
KV_SLOTS = 9.13 * 131072
PROMPT = 3000          # typical prompt tokens incl. chat template
W = 21600              # hidden-run LLM window (s), ~6 h after HA (v2 design)
ANS = 0                # v2 behaviour: no forced answer tokens


def plan(d, cap, conc):
    pstop = d.cdf(cap)
    e_len = integrate.quad(lambda x: x * d.pdf(x), 0, cap, limit=200)[0] + cap * (1 - pstop)
    dur = e_len / SPEED
    slots = conc * W - conc * 0.5 * dur            # last wave: on average half a job per slot cannot start/finish
    jobs = slots / dur
    ver = jobs * (pstop + (1 - pstop) * v_trunc)
    kv = conc * (cap + PROMPT) / KV_SLOTS
    return dict(cap=cap, conc=conc, p_stop=round(pstop, 3), dur_s=round(dur), jobs=round(jobs, 1), verified=round(ver, 1),
                kv_peak=round(kv, 2))


fits = {"lognormal": fit_logn(obs, cens), "weibull": fit_weib(obs, cens), "lognormal_p10": boot_low()}
print(f"stops {len(obs)} censored {cens}; truncated-draft verify rate {v_trunc:.3f}")
GAIN_PER_VER = 0.23    # E3: (0.167 + 2.833) task-score / 13 verified programs on keys30
out = {}
for name, d in fits.items():
    rows = [plan(d, 64000, 13), plan(d, 80000, 13), plan(d, 96000, 11), plan(d, 96000, 12), plan(d, 120000, 9)]
    base = rows[0]["verified"]
    for r in rows:
        r["d_verified_vs_64k"] = round(r["verified"] - base, 1)
        r["d_public_pp_est"] = round((r["verified"] - base) * GAIN_PER_VER * 100 / 120, 2)
    out[name] = rows
    print(name)
    for r in rows:
        print("  ", r)
json.dump({"inputs": {"stops": obs.tolist(), "censored_at_64k": cens, "v_trunc": v_trunc, "speed": SPEED, "window_s": W,
                      "kv_slots": KV_SLOTS, "gain_per_verified_tasks": GAIN_PER_VER}, "plans": out},
          open(f"{root}/artifacts/verification/arc-agi-2/v13/cap-yield-v1.json", "w"), indent=1)


# --- stratified model (advisor 10-15): all 12 stops come from 7 "reachable" tasks; the other tasks never stopped ---
rows_all = []
for name in ("v13-e3-seeds-v1", "v13-progsynth-v2"):
    for l in open(f"{root}/runs/arc-agi-2/v6/output-arc-2026-{name}/v13-programs.jsonl"):
        r = json.loads(l) if l.strip() else None
        if r and r["state"] == "done":
            rows_all.append(r)
reach = {r["task"] for r in rows_all if r["finish_reason"] == "stop"}
R = [r for r in rows_all if r["task"] in reach]
N = [r for r in rows_all if r["task"] not in reach]
w_r = len(R) / len(rows_all)
o_r = np.array([int(r["completion_tokens"]) for r in R if r["finish_reason"] == "stop"], float)
c_r = sum(r["finish_reason"] == "length" for r in R)
vt_r = sum(bool(r.get("verified")) for r in R if r["finish_reason"] == "length") / max(1, c_r)
vt_n = sum(bool(r.get("verified")) for r in N) / len(N)
d_r = fit_logn(o_r, c_r)
print(f"\nstratified: reachable tasks {len(reach)} jobs {len(R)} (w {w_r:.2f}) stops {len(o_r)} trunc-verify {vt_r:.2f}; "
      f"never-stopped jobs {len(N)} trunc-verify {vt_n:.3f}; reachable P(stop<=64k) {d_r.cdf(64000):.2f} "
      f"<=80k {d_r.cdf(80000):.2f} <=96k {d_r.cdf(96000):.2f}")


def plan_strat(cap, conc, q, reach_frozen=False):
    """q = P(natural stop <= cap | never-stopped stratum); those stops assumed spread uniformly over (64k, cap].
    reach_frozen: the reachable stratum gains no stops past 64k (sensitivity: no lognormal extrapolation; Argon P1-2)."""
    pr = d_r.cdf(min(cap, 64000) if reach_frozen else cap)
    len_r = integrate.quad(lambda x: x * d_r.pdf(x), 0, min(cap, 64000) if reach_frozen else cap, limit=200)[0] \
        + cap * (1 - pr)
    len_n = q * (64000 + cap) / 2 + (1 - q) * cap if cap > 64000 else cap
    e_len = w_r * len_r + (1 - w_r) * len_n
    dur = e_len / SPEED
    jobs = (conc * W - conc * 0.5 * dur) / dur
    per_job = w_r * (pr + (1 - pr) * vt_r) + (1 - w_r) * (q + (1 - q) * vt_n)
    return dict(cap=cap, conc=conc, q_never=q, dur_s=round(dur), jobs=round(jobs, 1), per_job=round(per_job, 3),
                verified=round(jobs * per_job, 1))


strat = {}
base = plan_strat(64000, 13, 0.0)
print("  ", base)
for cap, conc in ((80000, 11), (96000, 10), (96000, 12)):
    for q in (0.0, 0.05, 0.15, 0.3):
        p = plan_strat(cap, conc, q)
        p["d_verified_vs_64k"] = round(p["verified"] - base["verified"], 1)
        p["d_public_pp_range"] = [round(p["d_verified_vs_64k"] * g * 100 / 120, 2) for g in (0.04, 0.23, 0.35)]
        strat[f"{cap}-c{conc}-q{q}"] = p
        print("  ", p)
res = json.load(open(f"{root}/artifacts/verification/arc-agi-2/v13/cap-yield-v1.json"))
res["stratified"] = {"reachable_tasks": sorted(reach), "w_reachable": w_r, "trunc_verify": {"reachable": vt_r, "never": vt_n},
                     "reachable_fit_cdf": {c: round(float(d_r.cdf(c)), 3) for c in (64000, 80000, 96000)},
                     "base_64k_c13": base, "plans": strat,
                     "note": "conc from v13_llm formula int(kv/(0.35 cap + 6000)) + 2 with kv 331,200: 64k->13, 80k->11, "
                             "96k->10; c12 at 96k is an override (KV est. 0.99). gain/verified range 0.04-0.35 task-score "
                             "(E3 rep 1 0.167/5 = 0.03, pooled 3.0/13 = 0.23, rep 2 2.83/8 = 0.35)."}
json.dump(res, open(f"{root}/artifacts/verification/arc-agi-2/v13/cap-yield-v1.json", "w"), indent=1)

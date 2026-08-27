#!/usr/bin/env python3
"""
M2: is DDS's lower effective dimension an artifact of within-pool dependence?

The DDS pool is 8 agents x 3 rounds x 5 trials = 120 responses, correlated within
a trial (rounds accumulate) and within a lineage. The participation ratio grows
with sample count, and correlated samples carry fewer *effective* samples, so
DDS's ~0.95x d_eff could be a statistical consequence of autocorrelation rather
than a property of the closed loop.

We test this by DECORRELATING the DDS pool and checking whether its effective
dimension jumps up toward the reference:

  full pool     : all 120 responses (rounds accumulated -- most correlated)
  final round   : only the last round, 8 agents x 5 trials = 40 (no round-to-round
                  accumulation -- much less correlated)
  per trial     : d_eff computed within each trial (24 responses) then averaged
                  (no cross-trial pooling)

Each DDS variant is compared to an independent reference subsampled to the SAME n.
If the DDS/independent ratio stays near 1 (well below "clears the reference") as we
decorrelate, the confinement is not a dependence artifact. The subspace-leakage
test (analyze_subspace_leakage.py) already gives a dependence-robust check, since
the captured-fraction metric is not monotone in n the way the participation ratio
is; this script confirms the point for the effective dimension itself.
"""

import sys, os, json, glob
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DYN = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
EMB = "all-MiniLM-L6-v2"
B = 2000
SEED = 0


def load_reference():
    f = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))[-1]
    return {t: [r["text"] for r in rs]
            for t, rs in json.load(open(f))["responses_by_task"].items()}


def load_dds_variants():
    """returns {task: {'full': [...], 'final': [...], 'trials': [[...], ...]}}"""
    d = json.load(open(DYN)); out = {}
    for td in d["conditions"]["dds_alpha_0.5"]:
        t = td["task_id"]; o = out.setdefault(t, {"full": [], "final": [], "trials": []})
        for tr in td["trials"]:
            rts = tr.get("response_texts", [])
            trial_texts = []
            for rnd in rts:
                trial_texts.extend(r["text"] for r in rnd)
            if trial_texts:
                o["full"].extend(trial_texts)
                o["trials"].append(trial_texts)
            if rts:
                o["final"].extend(r["text"] for r in rts[-1])   # last round only
    return out


def pr(E):
    if len(E) < 3:
        return float("nan")
    ev = PCA(n_components=min(len(E), E.shape[1])).fit(E).explained_variance_
    return float((ev.sum() ** 2) / np.square(ev).sum())


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB)
    enc = lambda ts: np.asarray(model.encode(ts, show_progress_bar=False), dtype=float)

    ref = load_reference(); dds = load_dds_variants()
    tasks = list(ref.keys())
    Eref = {t: enc(ref[t]) for t in tasks}
    Efull = {t: enc(dds[t]["full"]) for t in tasks}
    Efinal = {t: enc(dds[t]["final"]) for t in tasks}
    Etrials = {t: [enc(tt) for tt in dds[t]["trials"]] for t in tasks}
    rng = np.random.RandomState(SEED)
    print(f"embedding = {EMB}; B={B}\n")

    def ref_mean_ci(n):
        r2 = np.random.RandomState(SEED + 1)
        pts = [pr(Eref[t][r2.choice(len(Eref[t]), n, replace=False)]) for t in tasks]
        draws = np.array([np.mean([pr(Eref[t][r2.randint(0, len(Eref[t]), n)])
                                   for t in tasks]) for _ in range(B)])
        return float(np.mean(pts)), float(draws.std(ddof=1))

    def dds_mean_ci(Emap, n=None):
        r2 = np.random.RandomState(SEED + 2)
        if n is None:                          # use full arrays as-is
            pts = [pr(Emap[t]) for t in tasks]
            draws = np.array([np.mean([pr(Emap[t][r2.randint(0, len(Emap[t]), len(Emap[t]))])
                                       for t in tasks]) for _ in range(B)])
        else:
            pts = [pr(Emap[t][r2.choice(len(Emap[t]), n, replace=False)]) for t in tasks]
            draws = np.array([np.mean([pr(Emap[t][r2.randint(0, len(Emap[t]), n)])
                                       for t in tasks]) for _ in range(B)])
        return float(np.mean(pts)), float(draws.std(ddof=1))

    def trial_mean_ci():
        # d_eff within each trial (n=24), averaged over trials then tasks
        pts = [np.mean([pr(E) for E in Etrials[t]]) for t in tasks]
        r2 = np.random.RandomState(SEED + 3)
        draws = []
        for _ in range(B):
            v = []
            for t in tasks:
                Es = Etrials[t]
                v.append(np.mean([pr(E[r2.randint(0, len(E), len(E))]) for E in Es]))
            draws.append(np.mean(v))
        return float(np.mean(pts)), float(np.std(draws, ddof=1))

    def line(label, n, dpt, dse, rpt, rse):
        ratio = dpt / rpt
        dlo, dhi = dpt - 1.96 * dse, dpt + 1.96 * dse
        rlo, rhi = rpt - 1.96 * rse, rpt + 1.96 * rse
        clears = dlo > rhi
        v = "CLEARS ref" if clears else ("overlaps ref" if not (dhi < rlo) else "below ref")
        print(f"{label:<26} n={n:<4} DDS {dpt:5.1f} [{dlo:4.1f},{dhi:4.1f}]  "
              f"ref {rpt:5.1f} [{rlo:4.1f},{rhi:4.1f}]  ratio {ratio:.2f}x  {v}")
        return {"n": n, "dds": dpt, "dds_se": dse, "ref": rpt, "ref_se": rse,
                "ratio": ratio, "clears": clears}

    out = {}
    rp120, rs120 = ref_mean_ci(120)
    dp, ds = dds_mean_ci(Efull)
    out["full_pool"] = line("full pool (accumulated)", 120, dp, ds, rp120, rs120)

    rp40, rs40 = ref_mean_ci(40)
    dp, ds = dds_mean_ci(Efinal)
    out["final_round"] = line("final round only", 40, dp, ds, rp40, rs40)

    rp24, rs24 = ref_mean_ci(24)
    dp, ds = trial_mean_ci()
    out["per_trial"] = line("per trial (within-trial)", 24, dp, ds, rp24, rs24)

    Path("results/dynamics_mapelites/m2_dependence.json").write_text(json.dumps(out, indent=2))
    print("\nSaved: results/dynamics_mapelites/m2_dependence.json")
    print("Read: if the DDS/reference ratio stays near 1 (never CLEARS) as the pool")
    print("is decorrelated (full -> final-round-only -> per-trial), the confinement")
    print("is not an artifact of within-pool dependence.")


if __name__ == "__main__":
    main()

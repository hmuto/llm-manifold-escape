#!/usr/bin/env python3
"""
Does the dimensional ceiling hold across the SELECTION FAMILY, or only for DDS?

The main text measures the participation-ratio effective dimension for DDS
(0.92x the reference) and for the decoding/prompt levers. It does not measure it
for the other selection-based conditions. Here we compute it for the multi-round
selection conditions in the dynamics experiment (DDS alpha=0.5, MAP-Elites,
Debate) and compare each to the independent-128 reference, with bootstrap CIs.

If MAP-Elites and Debate also hold the effective dimension near the reference
(like DDS), the ceiling is a property of closed-loop selection generally, not of
DDS specifically. This strengthens the limit claim from one method + a conceptual
argument to several methods, empirically.

point estimate = participation ratio of the full embedded pool.
CI = point +/- 1.96 x bootstrap SE (native n, B resamples with replacement).
"""

import sys, os, json, glob
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DYN = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
B = 2000
SEED = 0
CONDS = ["dds_alpha_0.5", "map_elites", "debate", "independent"]
LABELS = {"dds_alpha_0.5": "DDS (a=0.5)", "map_elites": "MAP-Elites",
          "debate": "Debate", "independent": "Independent (dyn.)"}


def load_reference():
    f = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))[-1]
    d = json.load(open(f))
    return {tid: [r["text"] for r in rs] for tid, rs in d["responses_by_task"].items()}


def load_condition(cond):
    d = json.load(open(DYN))
    pool = {}
    for td in d["conditions"][cond]:
        tid = td["task_id"]; pool.setdefault(tid, [])
        for tr in td["trials"]:
            for rt in tr.get("response_texts", []):
                for r in rt:
                    pool[tid].append(r["text"])
    return pool


def participation_ratio(E):
    if len(E) < 3:
        return float("nan")
    ev = PCA(n_components=min(len(E), E.shape[1])).fit(E).explained_variance_
    return float((ev.sum() ** 2) / np.square(ev).sum())


def boot_se(E, rng):
    n = len(E)
    pr = np.array([participation_ratio(E[rng.randint(0, n, n)]) for _ in range(B)])
    return float(pr.std(ddof=1))


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda ts: np.asarray(model.encode(ts, show_progress_bar=False), dtype=float)

    ref = load_reference()
    tasks = list(ref.keys())
    Eref = {t: enc(ref[t]) for t in tasks}
    Econd = {c: {t: enc(texts[t]) for t in tasks}
             for c in CONDS for texts in [load_condition(c)]}

    rng = np.random.RandomState(SEED)
    print(f"point +/- 1.96 x bootstrap-SE (native n), B={B}")
    print(f"n/task: reference={np.mean([len(Eref[t]) for t in tasks]):.0f}, " +
          ", ".join(f"{LABELS[c]}={np.mean([len(Econd[c][t]) for t in tasks]):.0f}" for c in CONDS))
    print()

    # reference d_eff
    ref_pt = {t: participation_ratio(Eref[t]) for t in tasks}
    ref_mean = float(np.mean([ref_pt[t] for t in tasks]))

    header = f"{'task':<12} {'Reference':>18} " + " ".join(f"{LABELS[c]:>20}" for c in CONDS)
    print(header)
    table = {c: {} for c in CONDS}
    for t in tasks:
        cells = [f"{ref_pt[t]:5.1f} [{ref_pt[t]-1.96*boot_se(Eref[t],rng):4.1f},{ref_pt[t]+1.96*boot_se(Eref[t],rng):4.1f}]"]
        for c in CONDS:
            pt = participation_ratio(Econd[c][t]); se = boot_se(Econd[c][t], rng)
            table[c][t] = {"point": pt, "se": se}
            cells.append(f"{pt:5.1f} [{pt-1.96*se:4.1f},{pt+1.96*se:4.1f}]")
        print(f"{t:<12} {cells[0]:>18} " + " ".join(f"{x:>20}" for x in cells[1:]))

    # task-mean with bootstrap CI on the mean + ratio to reference
    print(f"\n{'MEAN':<12} {'Reference':>18} " + " ".join(f"{LABELS[c]:>20}" for c in CONDS))
    def mean_ci(Emap):
        r2 = np.random.RandomState(SEED + 1)
        draws = np.array([np.mean([participation_ratio(Emap[t][r2.randint(0, len(Emap[t]), len(Emap[t]))])
                                   for t in tasks]) for _ in range(B)])
        pt = float(np.mean([participation_ratio(Emap[t]) for t in tasks]))
        se = float(draws.std(ddof=1))
        return pt, se
    ref_m_pt, ref_m_se = mean_ci(Eref)
    cells = [f"{ref_m_pt:5.1f} [{ref_m_pt-1.96*ref_m_se:4.1f},{ref_m_pt+1.96*ref_m_se:4.1f}]"]
    means = {"reference": {"point": ref_m_pt, "se": ref_m_se, "ratio": 1.0}}
    for c in CONDS:
        pt, se = mean_ci(Econd[c])
        means[c] = {"point": pt, "se": se, "ratio": pt / ref_m_pt}
        cells.append(f"{pt:5.1f} [{pt-1.96*se:4.1f},{pt+1.96*se:4.1f}]")
    print(f"{'':<12} {cells[0]:>18} " + " ".join(f"{x:>20}" for x in cells[1:]))

    print("\n=== task-mean effective dimension, ratio to reference ===")
    print(f"  reference           {ref_m_pt:5.1f}   (1.00x)")
    for c in CONDS:
        print(f"  {LABELS[c]:<18}  {means[c]['point']:5.1f}   ({means[c]['ratio']:.2f}x)")

    out = Path("results/dynamics_mapelites/effdim_selection_family.json")
    json.dump({"B": B, "reference_mean": ref_m_pt, "means": means, "per_task": table},
              open(out, "w"), indent=2)
    print(f"\nSaved: {out}")
    print("\nRead: if MAP-Elites and Debate ratios are ~1 (like DDS 0.92x), the")
    print("dimensional ceiling holds across the selection family, not just DDS.")


if __name__ == "__main__":
    main()

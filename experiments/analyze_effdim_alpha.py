#!/usr/bin/env python3
"""
Does the effective dimension stay flat as selection pressure alpha varies?

The alpha sweep shows diversity is non-monotonic in the pressure (peak at ~0.5,
decline after). That is a DIVERSITY argument. Here we test the DIMENSIONAL claim
directly: does the participation-ratio effective dimension change with alpha? The
dynamics experiment saved response texts for DDS at alpha in {0.0, 0.5, 1.0}
(120 responses/task each), so we can compute d_eff across the rising edge and the
peak from existing data (no re-running). If d_eff stays near the independent-128
reference for all three, then raising the pressure does not add dimensions, which
ties the alpha sweep directly to the dimensional claim (not just to diversity).

point = participation ratio of the full embedded pool.
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
CONDS = ["dds_alpha_0.0", "dds_alpha_0.5", "dds_alpha_1.0"]
LABELS = {"dds_alpha_0.0": "DDS a=0.0", "dds_alpha_0.5": "DDS a=0.5",
          "dds_alpha_1.0": "DDS a=1.0"}


def load_reference():
    f = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))[-1]
    d = json.load(open(f))
    return {tid: [r["text"] for r in rs] for tid, rs in d["responses_by_task"].items()}


def load_condition(cond):
    d = json.load(open(DYN)); pool = {}
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

    ref = load_reference(); tasks = list(ref.keys())
    Eref = {t: enc(ref[t]) for t in tasks}
    Econd = {c: {t: enc(texts[t]) for t in tasks}
             for c in CONDS for texts in [load_condition(c)]}
    rng = np.random.RandomState(SEED)

    print(f"point +/- 1.96 x bootstrap-SE (native n), B={B}\n")
    print(f"{'task':<12} {'Reference':>18} " + " ".join(f"{LABELS[c]:>18}" for c in CONDS))
    for t in tasks:
        cells = [f"{participation_ratio(Eref[t]):5.1f} [{participation_ratio(Eref[t])-1.96*boot_se(Eref[t],rng):4.1f},{participation_ratio(Eref[t])+1.96*boot_se(Eref[t],rng):4.1f}]"]
        for c in CONDS:
            pt = participation_ratio(Econd[c][t]); se = boot_se(Econd[c][t], rng)
            cells.append(f"{pt:5.1f} [{pt-1.96*se:4.1f},{pt+1.96*se:4.1f}]")
        print(f"{t:<12} {cells[0]:>18} " + " ".join(f"{x:>18}" for x in cells[1:]))

    def mean_ci(Emap):
        r2 = np.random.RandomState(SEED + 1)
        draws = np.array([np.mean([participation_ratio(
            Emap[t][r2.randint(0, len(Emap[t]), len(Emap[t]))]) for t in tasks]) for _ in range(B)])
        return float(np.mean([participation_ratio(Emap[t]) for t in tasks])), float(draws.std(ddof=1))

    ref_pt, ref_se = mean_ci(Eref)
    print(f"\n{'MEAN':<12} {'Reference':>18} " + " ".join(f"{LABELS[c]:>18}" for c in CONDS))
    means = {"reference": {"point": ref_pt, "se": ref_se, "ratio": 1.0}}
    cells = [f"{ref_pt:5.1f} [{ref_pt-1.96*ref_se:4.1f},{ref_pt+1.96*ref_se:4.1f}]"]
    for c in CONDS:
        pt, se = mean_ci(Econd[c]); means[c] = {"point": pt, "se": se, "ratio": pt / ref_pt}
        cells.append(f"{pt:5.1f} [{pt-1.96*se:4.1f},{pt+1.96*se:4.1f}]")
    print(f"{'':<12} {cells[0]:>18} " + " ".join(f"{x:>18}" for x in cells[1:]))

    print("\n=== task-mean d_eff, ratio to reference ===")
    print(f"  reference     {ref_pt:5.1f}  (1.00x)")
    for c in CONDS:
        print(f"  {LABELS[c]:<12}  {means[c]['point']:5.1f}  ({means[c]['ratio']:.2f}x)  "
              f"{'overlaps ref' if not (means[c]['point']+1.96*means[c]['se'] < ref_pt-1.96*ref_se or means[c]['point']-1.96*means[c]['se'] > ref_pt+1.96*ref_se) else 'CLEARS ref'}")
    out = Path("results/dynamics_mapelites/effdim_alpha.json")
    json.dump({"B": B, "reference": {"point": ref_pt, "se": ref_se}, "means": means}, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")
    print("Read: if d_eff is flat (ratio ~1) across alpha=0.0/0.5/1.0, raising the")
    print("pressure does not add dimensions -> ties the alpha sweep to the dimensional claim.")


if __name__ == "__main__":
    main()

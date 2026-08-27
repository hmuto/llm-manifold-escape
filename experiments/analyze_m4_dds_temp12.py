#!/usr/bin/env python3
"""
M4 analysis: does selection keep or destroy the high-temperature dimensional gain?

We now have the missing factorial cell (DDS alpha=0.5 at decoding temperature 1.2).
Compare its effective dimension against the four reference points:

  independent T=0.7   (the low-temperature manifold, d_eff ~19)
  DDS T=0.7           (selection on the low-temperature manifold, ~18)
  independent T=1.2   (the widened manifold, ~26)
  DDS T=1.2           (selection on the widened manifold -- NEW)

If DDS T=1.2 stays near independent T=1.2, selection is confined *relative to its
decoding*: it neither adds nor destroys dimensions, so widening the decoding and
then selecting are complementary (selection harvests tails within the wider
manifold). If DDS T=1.2 falls back toward the T=0.7 level, selection re-confines
and antagonises the temperature gain. Matched n per task; bootstrap 95% CI.
"""

import sys, os, json, glob
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
B, SEED = 2000, 0


def load_independent(temp07=True):
    if temp07:
        f = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))[-1]
        return {t: [r["text"] for r in rs]
                for t, rs in json.load(open(f))["responses_by_task"].items()}
    f = sorted(glob.glob("results/temperature_expansion/temperature_expansion_2*.json"))[-1]
    d = json.load(open(f))["responses_by_temp_task"]["temp_1.2"]
    return {t: [r["text"] for r in tx] for t, tx in d.items()}


def load_dds(path, cond="dds_alpha_0.5"):
    d = json.load(open(path)); pool = {}
    for td in d["conditions"][cond]:
        t = td["task_id"]; pool.setdefault(t, [])
        for tr in td["trials"]:
            for rnd in tr.get("response_texts", []):
                pool[t].extend(r["text"] for r in rnd)
    return pool


def pr(E):
    if len(E) < 3:
        return float("nan")
    ev = PCA(n_components=min(len(E), E.shape[1])).fit(E).explained_variance_
    return float((ev.sum() ** 2) / np.square(ev).sum())


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda ts: np.asarray(model.encode(ts, show_progress_bar=False), dtype=float)

    dds12_file = sorted(glob.glob("results/temperature_expansion/dds_temp12_*.json"))[-1]
    dyn = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
    conds = {
        "independent T=0.7": load_independent(True),
        "DDS T=0.7": load_dds(dyn),
        "independent T=1.2": load_independent(False),
        "DDS T=1.2 (new)": load_dds(dds12_file),
    }
    tasks = list(conds["independent T=0.7"].keys())
    E = {c: {t: enc(conds[c][t]) for t in tasks} for c in conds}
    nmatch = {t: min(len(E[c][t]) for c in conds) for t in tasks}
    print(f"DDS T=1.2 file: {os.path.basename(dds12_file)}")
    print(f"matched n/task = { {t: nmatch[t] for t in tasks} }; B={B}\n")

    def mean_ci(c):
        r2 = np.random.RandomState(SEED + 1)
        pts = [pr(E[c][t][r2.choice(len(E[c][t]), nmatch[t], replace=False)]) for t in tasks]
        draws = np.array([np.mean([pr(E[c][t][r2.randint(0, len(E[c][t]), nmatch[t])])
                                   for t in tasks]) for _ in range(B)])
        return float(np.mean(pts)), float(draws.std(ddof=1))

    res = {c: mean_ci(c) for c in conds}
    ref07 = res["independent T=0.7"][0]; ref12 = res["independent T=1.2"][0]
    print(f"{'condition':<22} {'d_eff [95% CI]':>18} {'/T0.7':>7} {'/T1.2':>7}")
    out = {}
    for c in conds:
        pt, se = res[c]; lo, hi = pt - 1.96 * se, pt + 1.96 * se
        out[c] = {"point": pt, "se": se, "ratio_T07": pt / ref07, "ratio_T12": pt / ref12}
        print(f"{c:<22} {f'{pt:.1f} [{lo:.1f},{hi:.1f}]':>18} {pt/ref07:>6.2f}x {pt/ref12:>6.2f}x")

    d12 = res["DDS T=1.2 (new)"]; lo, hi = d12[0]-1.96*d12[1], d12[0]+1.96*d12[1]
    i12lo, i12hi = ref12-1.96*res["independent T=1.2"][1], ref12+1.96*res["independent T=1.2"][1]
    i07lo, i07hi = ref07-1.96*res["independent T=0.7"][1], ref07+1.96*res["independent T=0.7"][1]
    holds12 = not (hi < i12lo or lo > i12hi)
    above07 = lo > i07hi
    print(f"\nDDS T=1.2 overlaps independent T=1.2? {holds12}  (holds the widened dimension)")
    print(f"DDS T=1.2 clears independent T=0.7? {above07}  (keeps the temperature gain)")
    verdict = ("COMPLEMENTARY: selection holds d_eff at the widened T=1.2 manifold"
               if holds12 and above07 else
               "RE-CONFINES: selection pulls d_eff back toward T=0.7"
               if not above07 else "PARTIAL")
    print(f"=> {verdict}")

    Path("results/temperature_expansion/m4_dds_temp12.json").write_text(
        json.dumps({"conds": out, "holds_T12": holds12, "above_T07": above07,
                    "verdict": verdict}, indent=2))
    print("\nSaved: results/temperature_expansion/m4_dds_temp12.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Does decoding exceed the fixed-decoding support that selection cannot?

Reference support = independent N=128 at temperature 0.7 (the same estimate used
in analyze_support_vs_loop.py). For each candidate set we measure, against this
support:
  - escape fraction: fraction of candidate points that fall OUTSIDE the support
    (min cosine distance to the support > eps), eps = 2x median nearest-neighbour
    distance within the support (same eps as the support analysis).
  - radius: mean distance to the support centroid (extent).
  - effective dimension (participation ratio).

Candidates:
  - temp-0.7 self  (leave-one-out within the reference)  -> in-distribution baseline
  - DDS pool       (alpha=0.5, temperature 0.7; selection) -> should NOT escape
  - independent    (temperature 1.0)                       -> decoding lever
  - independent    (temperature 1.2)                       -> decoding lever

If DDS (selection at fixed decoding) does not escape but temperature 1.0/1.2 do,
then the ceiling is a property of the decoding, not immovable: selection cannot
exceed it, but decoding can.
"""

import sys, os, json, glob
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DDS_FILE = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"


def load_ref_temp07():
    f = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))[-1]
    d = json.load(open(f))
    return {tid: [r["text"] for r in rs] for tid, rs in d["responses_by_task"].items()}


def load_temp_expansion():
    f = sorted(glob.glob("results/temperature_expansion/temperature_expansion_*.json"))[-1]
    d = json.load(open(f))
    return d["responses_by_temp_task"]   # {"temp_1.0": {tid: [{text}]}, ...}


def load_dds_pool():
    d = json.load(open(DDS_FILE))
    pool = {}
    for td in d["conditions"]["dds_alpha_0.5"]:
        tid = td["task_id"]; pool.setdefault(tid, [])
        for trial in td["trials"]:
            for rt in trial.get("response_texts", []):
                for resp in rt:
                    pool[tid].append(resp["text"])
    return pool


def participation_ratio(embs):
    from sklearn.decomposition import PCA
    if len(embs) < 3:
        return float("nan")
    p = PCA(n_components=min(len(embs), embs.shape[1])).fit(embs)
    ev = p.explained_variance_
    return float((ev.sum() ** 2) / np.square(ev).sum())


def escape_fraction(cand, support, eps, leave_one_out=False):
    d = cosine_distances(cand, support)
    if leave_one_out:
        np.fill_diagonal(d, np.inf)
    return float(np.mean(d.min(axis=1) > eps))


def radius(embs, centroid):
    return float(np.mean(np.linalg.norm(embs - centroid, axis=1)))


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    ref = load_ref_temp07()
    temp_exp = load_temp_expansion()
    dds = load_dds_pool()
    tasks = list(ref.keys())

    rows = {k: {"escape": [], "radius_ratio": [], "eff": []}
            for k in ["temp07_self", "dds_temp07", "temp_1.0", "temp_1.2"]}

    print(f"{'task':<12} {'set':<12} {'escape%':>8} {'radiusX':>8} {'eff_dim':>8}")
    print("-" * 52)
    for tid in tasks:
        R = np.asarray(model.encode(ref[tid], show_progress_bar=False))
        centroid = R.mean(axis=0)
        rd = cosine_distances(R); np.fill_diagonal(rd, np.inf)
        eps = float(np.median(rd.min(axis=1))) * 2.0
        rad_ref = radius(R, centroid)

        sets = {
            "temp07_self": R,
            "dds_temp07": np.asarray(model.encode(dds[tid], show_progress_bar=False)),
            "temp_1.0": np.asarray(model.encode(
                [r["text"] for r in temp_exp["temp_1.0"][tid]], show_progress_bar=False)),
            "temp_1.2": np.asarray(model.encode(
                [r["text"] for r in temp_exp["temp_1.2"][tid]], show_progress_bar=False)),
        }
        for name, E in sets.items():
            esc = escape_fraction(E, R, eps, leave_one_out=(name == "temp07_self"))
            rr = radius(E, centroid) / rad_ref
            eff = participation_ratio(E)
            rows[name]["escape"].append(esc)
            rows[name]["radius_ratio"].append(rr)
            rows[name]["eff"].append(eff)
            print(f"{tid:<12} {name:<12} {esc*100:>7.1f}% {rr:>7.2f}x {eff:>8.1f}")
        print("-" * 52)

    print(f"\n{'MEAN over tasks':<12}")
    print(f"{'set':<14} {'escape%':>8} {'radiusX':>8} {'eff_dim':>8}")
    print("-" * 44)
    summary = {}
    for name, v in rows.items():
        e, r, f = np.mean(v["escape"]), np.mean(v["radius_ratio"]), np.mean(v["eff"])
        summary[name] = {"escape": e, "radius_ratio": r, "eff_dim": f}
        print(f"{name:<14} {e*100:>7.1f}% {r:>7.2f}x {f:>8.1f}")

    out = Path("results/temperature_expansion/temperature_expansion_analysis.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")
    print("\nRead: selection (dds_temp07) escape ~ baseline; temp 1.0/1.2 escape >> baseline")
    print("      => decoding reaches outside the temperature-0.7 support; selection does not.")


if __name__ == "__main__":
    main()

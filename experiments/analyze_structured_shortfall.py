#!/usr/bin/env python3
"""
Characterize the mild loop confinement on the structured task.

The matched-k analysis leaves a persistent 9-11 point shortfall on problem_1 (a
distance-rate word problem with a single correct answer). We ask two questions:
  (1) At which round does the shortfall appear? -> round-wise cumulative coverage
      of the independent-128 reference by the DDS pool.
  (2) Which regions are missed? -> compare the distance-to-centroid (how
      peripheral a response is) of covered vs uncovered reference points.

Open-ended tasks (creative_1) are included as a contrast.
"""

import json, glob
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

DDS_FILE = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"


def load_reference():
    f = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))[-1]
    d = json.load(open(f))
    return {tid: [r["text"] for r in rs] for tid, rs in d["responses_by_task"].items()}


def load_dds_by_round():
    d = json.load(open(DDS_FILE))
    by_round = {}
    for td in d["conditions"]["dds_alpha_0.5"]:
        tid = td["task_id"]; by_round.setdefault(tid, {0: [], 1: [], 2: []})
        for trial in td["trials"]:
            for ri, round_texts in enumerate(trial.get("response_texts", [])):
                if ri in by_round[tid]:
                    by_round[tid][ri].extend(r["text"] for r in round_texts)
    return by_round


def coverage_mask(pool_emb, ref_emb, eps):
    d = cosine_distances(ref_emb, pool_emb)
    return d.min(axis=1) < eps    # boolean per reference point


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ref_texts = load_reference()
    dds = load_dds_by_round()

    print(f"{'task':<12} {'cov R0':>7} {'cov R0-1':>9} {'cov R0-2':>9} "
          f"{'radius covered':>15} {'radius missed':>14}")
    print("-" * 72)
    out = {}
    for tid in ref_texts:
        R = np.asarray(model.encode(ref_texts[tid], show_progress_bar=False))
        centroid = R.mean(0)
        rd = cosine_distances(R); np.fill_diagonal(rd, np.inf)
        eps = float(np.median(rd.min(axis=1))) * 2.0

        covs = []
        cum = []
        for r in range(3):
            cum += dds[tid][r]
            E = np.asarray(model.encode(cum, show_progress_bar=False))
            covs.append(float(coverage_mask(E, R, eps).mean()))
        # missed-region characterization on the full pool (rounds 0-2)
        E_full = np.asarray(model.encode(cum, show_progress_bar=False))
        mask = coverage_mask(E_full, R, eps)
        rad = np.linalg.norm(R - centroid, axis=1)
        rad_cov = float(rad[mask].mean()) if mask.any() else float("nan")
        rad_miss = float(rad[~mask].mean()) if (~mask).any() else float("nan")
        out[tid] = {"cov_by_round": covs, "radius_covered": rad_cov,
                    "radius_missed": rad_miss, "n_missed": int((~mask).sum())}
        print(f"{tid:<12} {covs[0]:>6.1%} {covs[1]:>8.1%} {covs[2]:>8.1%} "
              f"{rad_cov:>15.3f} {rad_miss:>14.3f}")

    json.dump(out, open("results/support_vs_loop/structured_shortfall.json", "w"), indent=2)
    print("\nSaved: results/support_vs_loop/structured_shortfall.json")
    print("Read: for problem_1, coverage growth stalls and missed points sit farther")
    print("from the centroid (the loop misses the periphery of the structured task).")


if __name__ == "__main__":
    main()

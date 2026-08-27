#!/usr/bin/env python3
"""
Matched-k sweep: does 'selection does not exceed the generator' hold across
sample sizes?

Extends the support-vs-loop analysis (analyze_support_vs_loop.py) from a single
matched size (k=24) to k in {24, 48, 72, 96}, using EXISTING data only:
  - DDS alpha=0.5 pooled responses (from dynamics_mapelites; 120 per task)
  - Independent N=128 responses (from independent_scaling)

For each k and task, we compare coverage of the independent-128 reference by:
  - a random k-subset of the DDS pool
  - a random k-subset of the independent pool
averaged over resamples. If DDS tracks (or falls below) independent at every k,
the 'selection reaches but does not exceed the generator' conclusion is not an
artifact of one sample size.
"""

import sys
import os
import json
import glob
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DDS_FILE = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
KS = [24, 48, 72, 96]
N_RESAMPLE = 50


def load_independent():
    files = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))
    with open(files[-1]) as f:
        d = json.load(f)
    return {tid: [r["text"] for r in rs] for tid, rs in d["responses_by_task"].items()}


def load_dds_pool():
    with open(DDS_FILE) as f:
        d = json.load(f)
    pool = {}
    for td in d["conditions"]["dds_alpha_0.5"]:
        tid = td["task_id"]
        pool.setdefault(tid, [])
        for trial in td["trials"]:
            for rt in trial.get("response_texts", []):
                for resp in rt:
                    pool[tid].append(resp["text"])
    return pool


def coverage(pool_embs, ref_embs, eps):
    d = cosine_distances(ref_embs, pool_embs)
    return float(np.mean(d.min(axis=1) < eps))


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    indep_texts = load_independent()
    dds_texts = load_dds_pool()
    tasks = list(indep_texts.keys())

    results = {"ks": KS, "per_task": {}}
    print(f"Matched-k sweep, k in {KS}, {N_RESAMPLE} resamples per (task,k)\n")

    for tid in tasks:
        ind = np.asarray(model.encode(indep_texts[tid], show_progress_bar=False))
        dds = np.asarray(model.encode(dds_texts[tid], show_progress_bar=False))
        ref = ind
        # eps = 2x median NN distance within reference (same as support analysis)
        refd = cosine_distances(ref)
        np.fill_diagonal(refd, np.inf)
        eps = float(np.median(refd.min(axis=1))) * 2.0

        row = {}
        for k in KS:
            if k > len(dds) or k > len(ind):
                continue
            cov_dds, cov_ind = [], []
            for s in range(N_RESAMPLE):
                rng = np.random.RandomState(s)
                di = rng.choice(len(dds), size=k, replace=False)
                ii = rng.choice(len(ind), size=k, replace=False)
                cov_dds.append(coverage(dds[di], ref, eps))
                cov_ind.append(coverage(ind[ii], ref, eps))
            row[k] = {
                "cov_dds": float(np.mean(cov_dds)),
                "cov_ind": float(np.mean(cov_ind)),
                "diff": float(np.mean(cov_dds) - np.mean(cov_ind)),
            }
        results["per_task"][tid] = {"n_dds": len(dds), "n_ind": len(ind), "eps": eps, "by_k": row}

        print(f"[{tid}] n_dds={len(dds)}, n_ind={len(ind)}")
        for k in KS:
            if k in row:
                r = row[k]
                print(f"  k={k:3d}: DDS={r['cov_dds']:.1%}  indep={r['cov_ind']:.1%}  "
                      f"diff={r['diff']*100:+.1f} pts")

    # Aggregate across tasks per k
    print("\n" + "=" * 60)
    print("MEAN over tasks (DDS - independent coverage), by k")
    print("=" * 60)
    agg = {}
    for k in KS:
        diffs = [results["per_task"][t]["by_k"][k]["diff"]
                 for t in tasks if k in results["per_task"][t]["by_k"]]
        dds_m = [results["per_task"][t]["by_k"][k]["cov_dds"]
                 for t in tasks if k in results["per_task"][t]["by_k"]]
        ind_m = [results["per_task"][t]["by_k"][k]["cov_ind"]
                 for t in tasks if k in results["per_task"][t]["by_k"]]
        agg[k] = {"diff_mean": float(np.mean(diffs)), "diff_sd": float(np.std(diffs, ddof=1)),
                  "dds_mean": float(np.mean(dds_m)), "ind_mean": float(np.mean(ind_m))}
        print(f"  k={k:3d}: DDS={np.mean(dds_m):.1%}  indep={np.mean(ind_m):.1%}  "
              f"diff={np.mean(diffs)*100:+.1f} pts (sd {np.std(diffs, ddof=1)*100:.1f})")
    results["aggregate"] = agg

    out = Path("results/support_vs_loop")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "matched_k_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out/'matched_k_sweep.json'}")


if __name__ == "__main__":
    main()

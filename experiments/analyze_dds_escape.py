#!/usr/bin/env python3
"""
M2: does DDS generate points OUTSIDE the independent-128 support more than a
matched independent sample does?

Coverage (fraction of reference points near the pool) does not test this; it
measures the wrong direction. Here we measure, per task, the fraction of a pool's
points that fall OUTSIDE the reference support (min cosine distance to the
reference > eps), and compare DDS against a matched independent sample using the
SAME reference and the SAME eps (a fair, apples-to-apples baseline).

Design (split-half, to keep reference and test disjoint):
  R = half of the independent-128 (reference); eps = 2x median NN within R.
  escape_indep = fraction of the OTHER independent half that is > eps from R.
  escape_dds   = fraction of a matched-size DDS subset that is > eps from R.
Averaged over random splits. If escape_dds ~ escape_indep, DDS does not spill
outside the support beyond ordinary sampling variability (thesis holds). If
escape_dds >> escape_indep, DDS reaches outside the support (thesis threatened).
"""

import json, glob
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

DDS_FILE = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
N_SPLIT = 50


def load_ref():
    f = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))[-1]
    d = json.load(open(f))
    return {tid: [r["text"] for r in rs] for tid, rs in d["responses_by_task"].items()}


def load_dds():
    d = json.load(open(DDS_FILE)); pool = {}
    for td in d["conditions"]["dds_alpha_0.5"]:
        tid = td["task_id"]; pool.setdefault(tid, [])
        for tr in td["trials"]:
            for rt in tr.get("response_texts", []):
                for r in rt: pool[tid].append(r["text"])
    return pool


def escape(pool, ref, eps):
    return float(np.mean(cosine_distances(pool, ref).min(axis=1) > eps))


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ref_t, dds_t = load_ref(), load_dds()
    tasks = list(ref_t.keys())
    emb = lambda t: np.asarray(model.encode(t, show_progress_bar=False), dtype=float)

    print(f"{'task':<12} {'esc_indep%':>11} {'esc_DDS%':>10} {'diff(DDS-ind)':>14}")
    ei_all, ed_all = [], []
    for tid in tasks:
        I = emb(ref_t[tid]); D = emb(dds_t[tid])
        half = len(I) // 2
        eis, eds = [], []
        for s in range(N_SPLIT):
            rng = np.random.RandomState(s)
            perm = rng.permutation(len(I))
            R, T = I[perm[:half]], I[perm[half:]]
            rd = cosine_distances(R); np.fill_diagonal(rd, np.inf)
            eps = float(np.median(rd.min(axis=1))) * 2.0
            k = min(len(T), len(D))
            di = rng.choice(len(D), size=k, replace=False)
            eis.append(escape(T[:k], R, eps))
            eds.append(escape(D[di], R, eps))
        ei, ed = float(np.mean(eis)), float(np.mean(eds))
        ei_all.append(ei); ed_all.append(ed)
        print(f"{tid:<12} {ei*100:>10.1f}% {ed*100:>9.1f}% {(ed-ei)*100:>+13.1f}")
    print(f"{'mean':<12} {np.mean(ei_all)*100:>10.1f}% {np.mean(ed_all)*100:>9.1f}% "
          f"{(np.mean(ed_all)-np.mean(ei_all))*100:>+13.1f}")
    from scipy import stats
    t = stats.ttest_rel(ed_all, ei_all)
    print(f"\npaired t (DDS vs indep escape, n={len(tasks)} tasks): "
          f"t={t.statistic:.2f}, p={t.pvalue:.3f}")
    print("Interpretation: if diff ~ 0, DDS does not spill outside the support")
    print("beyond sampling variability; if diff >> 0, DDS reaches outside it.")


if __name__ == "__main__":
    main()

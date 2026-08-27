#!/usr/bin/env python3
"""
Mechanism test: subspace propagation.

Hypothesis for WHY the closed loop adds no dimensions: each round's context (the
survivors) carries information only in the subspace spanned by the previous
round's embeddings, so a later round's responses stay largely in
(previous-round subspace + model prior). If so, a later round's responses should
lie MORE within the previous round's principal subspace than a fresh independent
draw does.

Test, per task: fit a k-dim PCA subspace on round r's pooled responses, then
measure the fraction of variance of (a) round r+1 responses and (b) an
independent draw that this subspace captures (both centered on round r's mean).
If captured(round r+1) > captured(independent), later-round responses are more
confined to the previous round's subspace -> subspace propagation.
"""

import json, glob
import numpy as np
from sklearn.decomposition import PCA

DDS_FILE = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"


def load_dds_by_round():
    d = json.load(open(DDS_FILE)); by = {}
    for td in d["conditions"]["dds_alpha_0.5"]:
        t = td["task_id"]; by.setdefault(t, {0: [], 1: [], 2: []})
        for tr in td["trials"]:
            for ri, rt in enumerate(tr.get("response_texts", [])):
                if ri in by[t]:
                    by[t][ri].extend(r["text"] for r in rt)
    return by


def load_indep():
    f = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))[-1]
    d = json.load(open(f))
    return {t: [r["text"] for r in rs] for t, rs in d["responses_by_task"].items()}


def captured_frac(X, mean, basis):
    """Fraction of variance of X (centered on `mean`) captured by `basis` (k x d)."""
    Xc = X - mean
    proj = Xc @ basis.T                    # (n, k)
    return float((proj ** 2).sum() / (Xc ** 2).sum())


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dds, indep = load_dds_by_round(), load_indep()
    tasks = list(dds.keys())
    emb = lambda t: np.asarray(model.encode(t, show_progress_bar=False), dtype=float)

    print(f"{'task':<12} {'k':>3} {'cap(r+1|r)':>11} {'cap(indep|r)':>13} {'diff':>7}")
    diffs = []
    for tid in tasks:
        R0 = emb(dds[tid][0]); R1 = emb(dds[tid][1]); I = emb(indep[tid])
        # subspace = top-k PCA of round 0 capturing ~90% variance
        p = PCA().fit(R0)
        k = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.90) + 1)
        basis = p.components_[:k]; mean = R0.mean(0)
        c1 = captured_frac(R1, mean, basis)
        ci = captured_frac(I, mean, basis)
        diffs.append(c1 - ci)
        print(f"{tid:<12} {k:>3} {c1:>10.3f} {ci:>12.3f} {c1-ci:>+7.3f}")
    print(f"{'mean':<12} {'':>3} {'':>11} {'':>13} {np.mean(diffs):>+7.3f}")
    from scipy import stats
    t = stats.ttest_rel([d for d in diffs], [0]*len(diffs))
    print(f"\npaired sign of diff (n={len(tasks)} tasks): mean {np.mean(diffs):+.3f}")
    print("If diff > 0 on tasks: round r+1 is MORE confined to round r's subspace")
    print("than an independent draw => subspace propagation (mechanism supported).")


if __name__ == "__main__":
    main()

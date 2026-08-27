#!/usr/bin/env python3
"""
Support-vs-loop analysis: does the DDS regeneration loop reach the LLM's raw
output support, or is it confined to a sub-region?

Inputs:
  - Independent N=128 responses per task (raw output support estimate)
  - DDS alpha=0.5 responses per task (from dynamics_mapelites; the loop's reach)

Extensive (coverage) metrics, computed on all-MiniLM embeddings per task:
  1. Coverage-vs-N curve for INDEPENDENT (subsample 8,16,32,64,128): does the
     support keep growing with N, or saturate?
  2. Matched-N coverage: DDS pool (24) vs a random independent subset of the
     same size, measured as fraction of the independent-128 REFERENCE support
     each pool covers (a reference point is "covered" if within eps of the pool).
  3. Effective dimension (participation ratio) and radius (mean dist to the
     reference centroid) for independent-128 vs DDS pool.

Verdict:
  Pattern A: DDS coverage ~ matched independent coverage, and independent
             saturates by N=128 near the DDS level -> DDS reaches the support.
  Pattern B: independent coverage (esp. matched-N) exceeds DDS -> loop confines.
"""

import sys
import os
import json
import glob
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances

DDS_FILE = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"


def load_independent():
    files = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))
    if not files:
        raise SystemExit("No independent_scaling results found. Run run_independent_scaling.py first.")
    with open(files[-1]) as f:
        d = json.load(f)
    print(f"Independent file: {files[-1]}")
    return {tid: [r["text"] for r in rs] for tid, rs in d["responses_by_task"].items()}


def load_dds_pool():
    """Per-task DDS alpha=0.5 responses, pooled across trials (the loop's reach)."""
    with open(DDS_FILE) as f:
        d = json.load(f)
    pool = {}
    per_trial = {}
    for td in d["conditions"]["dds_alpha_0.5"]:
        tid = td["task_id"]
        pool.setdefault(tid, [])
        per_trial.setdefault(tid, [])
        for trial in td["trials"]:
            trial_texts = []
            for rt in trial.get("response_texts", []):
                for resp in rt:
                    pool[tid].append(resp["text"])
                    trial_texts.append(resp["text"])
            if trial_texts:
                per_trial[tid].append(trial_texts)
    return pool, per_trial


def embed(model, texts):
    return np.asarray(model.encode(texts, convert_to_numpy=True, show_progress_bar=False))


def participation_ratio(embs):
    if len(embs) < 3:
        return float("nan")
    from sklearn.decomposition import PCA
    p = PCA(n_components=min(len(embs), embs.shape[1]))
    p.fit(embs)
    ev = p.explained_variance_
    return float((ev.sum() ** 2) / np.square(ev).sum())


def radius(embs, centroid):
    return float(np.mean(np.linalg.norm(embs - centroid, axis=1)))


def coverage(pool_embs, ref_embs, eps):
    """Fraction of reference points within eps (cosine dist) of the pool."""
    d = cosine_distances(ref_embs, pool_embs)   # (n_ref, n_pool)
    min_d = d.min(axis=1)
    return float(np.mean(min_d < eps))


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    indep_texts = load_independent()
    dds_pool_texts, dds_per_trial = load_dds_pool()

    tasks = list(indep_texts.keys())
    summary = {}

    for tid in tasks:
        ind = embed(model, indep_texts[tid])              # ~128 independent
        dds = embed(model, dds_pool_texts[tid])           # pooled DDS (~120)
        ref = ind                                         # reference support = independent-128
        centroid = ref.mean(axis=0)

        # eps = median nearest-neighbor distance within the reference (natural scale)
        refd = cosine_distances(ref)
        np.fill_diagonal(refd, np.inf)
        eps = float(np.median(refd.min(axis=1))) * 2.0    # 2x NN as "covered" radius

        # (1) coverage-vs-N for independent (self-coverage grows to ~1 by construction;
        #     instead report effective dim + radius vs N, which are the extent measures)
        curve = []
        rng = np.random.RandomState(0)
        for n in [8, 16, 32, 64, min(128, len(ind))]:
            idx = rng.choice(len(ind), size=min(n, len(ind)), replace=False)
            sub = ind[idx]
            curve.append({"N": int(n),
                          "eff_dim": participation_ratio(sub),
                          "radius": radius(sub, centroid)})

        # (2) matched-N coverage of the reference: DDS vs random-independent, same N
        n_match = min(len(dds), len(ind))
        # use per-trial DDS pools (24 each) matched to random independent-24
        matched = []
        for trial_texts in dds_per_trial[tid]:
            d_emb = embed(model, trial_texts)
            k = len(d_emb)
            cov_dds = coverage(d_emb, ref, eps)
            # random independent subset of same size k
            covs_ind = []
            for s in range(20):
                idx = np.random.RandomState(s).choice(len(ind), size=min(k, len(ind)), replace=False)
                covs_ind.append(coverage(ind[idx], ref, eps))
            matched.append({"k": k, "cov_dds": cov_dds, "cov_ind_mean": float(np.mean(covs_ind))})

        # (3) extent of full DDS pool vs independent-128
        eff_ind = participation_ratio(ind)
        eff_dds = participation_ratio(dds)
        rad_ind = radius(ind, centroid)
        rad_dds = radius(dds, centroid)
        cov_full_dds = coverage(dds, ref, eps)

        summary[tid] = {
            "n_indep": len(ind), "n_dds_pool": len(dds), "eps": eps,
            "curve": curve,
            "matched": matched,
            "eff_ind": eff_ind, "eff_dds": eff_dds,
            "rad_ind": rad_ind, "rad_dds": rad_dds,
            "cov_full_dds": cov_full_dds,
        }

        print(f"\n[{tid}]  n_indep={len(ind)}, n_dds_pool={len(dds)}, eps={eps:.3f}")
        print(f"  eff_dim: independent-128={eff_ind:.1f} | DDS pool={eff_dds:.1f}")
        print(f"  radius : independent-128={rad_ind:.3f} | DDS pool={rad_dds:.3f}")
        print(f"  DDS full-pool coverage of reference: {cov_full_dds:.2%}")
        cd = np.mean([m["cov_dds"] for m in matched])
        ci = np.mean([m["cov_ind_mean"] for m in matched])
        print(f"  matched-N ({matched[0]['k']}) coverage: DDS={cd:.2%} vs random-indep={ci:.2%}")
        print(f"  independent eff_dim vs N: " +
              ", ".join(f"N{c['N']}={c['eff_dim']:.1f}" for c in curve))

    out = Path("results/support_vs_loop")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "support_vs_loop.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    # Aggregate matched-N coverage difference (DDS - random independent)
    diffs, eff_ratios, rad_ratios = [], [], []
    for tid, s in summary.items():
        cd = np.mean([m["cov_dds"] for m in s["matched"]])
        ci = np.mean([m["cov_ind_mean"] for m in s["matched"]])
        diffs.append(cd - ci)
        eff_ratios.append(s["eff_dds"] / s["eff_ind"])
        rad_ratios.append(s["rad_dds"] / s["rad_ind"])
    print(f"Matched-N coverage (DDS - random-indep), mean over tasks: {np.mean(diffs):+.2%}")
    print(f"  (>0: DDS spreads better than random independent at same N)")
    print(f"  (<0: DDS confined below random independent -> loop trap signal)")
    print(f"eff_dim ratio DDS/independent-128, mean: {np.mean(eff_ratios):.2f}")
    print(f"radius ratio  DDS/independent-128, mean: {np.mean(rad_ratios):.2f}")
    print("\nInterpretation:")
    print("  Pattern A (DDS reaches support): eff/radius ratios ~1, matched coverage >= 0")
    print("  Pattern B (ICL trap): eff/radius ratios << 1, matched coverage < 0")


if __name__ == "__main__":
    main()

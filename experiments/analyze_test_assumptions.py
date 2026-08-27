#!/usr/bin/env python3
"""
Statistical-assumption checks for the headline paired comparisons.

For each headline contrast on cumulative (final) diversity from the dynamics
experiment, report:
  - paired t-test p (reproduces the value in the paper),
  - Shapiro-Wilk test on the paired differences (normality assumption of the
    paired t-test),
  - Wilcoxon signed-rank p (distribution-free robustness check).

Uses only the stored per-trial `final_diversity` values (no re-embedding).
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats

DATA = Path("results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json")


def final_divs(cond_data):
    vals = []
    for task_data in cond_data:
        for trial in task_data["trials"]:
            if "final_diversity" in trial:
                vals.append(trial["final_diversity"])
    return np.asarray(vals, dtype=float)


def cumulative_divs(cond_data, model):
    """Per-trial cumulative diversity: mean pairwise cosine distance over all
    responses pooled across rounds (re-embedded), matching the paper's metric."""
    from sklearn.metrics.pairwise import cosine_distances
    vals = []
    for task_data in cond_data:
        for trial in task_data["trials"]:
            texts = []
            for round_texts in trial.get("response_texts", []):
                for resp in round_texts:
                    texts.append(resp["text"])
            if len(texts) < 2:
                continue
            emb = model.encode(texts, show_progress_bar=False)
            dm = cosine_distances(emb)
            iu = np.triu_indices(len(texts), k=1)
            vals.append(float(dm[iu].mean()))
    return np.asarray(vals, dtype=float)


def report(name, a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = a - b
    t_p = stats.ttest_rel(a, b).pvalue
    sw_W, sw_p = stats.shapiro(diff)
    try:
        w_p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        w_p = float("nan")
    print(f"{name}")
    print(f"  n={n}  mean1={a.mean():.4f}  mean2={b.mean():.4f}")
    print(f"  paired t-test        p = {t_p:.5f}")
    print(f"  Shapiro-Wilk (diffs) W = {sw_W:.3f}, p = {sw_p:.3f}"
          f"   {'(normal OK)' if sw_p > 0.05 else '(non-normal)'}")
    print(f"  Wilcoxon signed-rank p = {w_p:.5f}")
    print()


def main():
    d = json.load(open(DATA))
    c = d["conditions"]
    dds05 = final_divs(c["dds_alpha_0.5"])
    dds00 = final_divs(c["dds_alpha_0.0"])
    dds10 = final_divs(c["dds_alpha_1.0"])
    me = final_divs(c["map_elites"])
    indep = final_divs(c["independent"])

    print("=" * 64)
    print("TEST-ASSUMPTION CHECKS (cumulative/final diversity, dynamics)")
    print("=" * 64)
    print(f"conditions: {list(c.keys())}\n")

    print("--- Snapshot (final-round) diversity ---\n")
    report("DDS a=0.5 vs MAP-Elites [snapshot]", dds05, me)

    print("--- Cumulative diversity (pooled across rounds; re-embedded) ---\n")
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    cum_dds05 = cumulative_divs(c["dds_alpha_0.5"], m)
    cum_me = cumulative_divs(c["map_elites"], m)
    cum_indep = cumulative_divs(c["independent"], m)
    report("DDS a=0.5 vs Independent [cumulative]", cum_dds05, cum_indep)
    report("MAP-Elites vs Independent [cumulative]", cum_me, cum_indep)
    report("DDS a=0.5 vs MAP-Elites [cumulative]", cum_dds05, cum_me)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cross-Judge Quality Evaluation

Re-evaluates response quality with Claude Haiku 4.5 (independent of GPT-4o-mini
used in the original G-Eval) to address self-preference bias concerns.

Inputs:
  - Original responses: results/dynamics_mapelites/dynamics_mapelites_*.json
  - Original GPT-4o-mini scores: results/quality_posthoc/quality_posthoc_*.json

Output:
  - Per-response Claude scores
  - Inter-judge agreement (Pearson r)
  - Re-run pairwise comparisons under Claude judge
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_agent import QualityEvaluator, EXAMPLE_TASKS

DYNAMICS = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
ORIG_QUALITY = "results/quality_posthoc/quality_posthoc_20260207_210944.json"
JUDGE_MODEL = "claude-haiku-4-5-20251001"


def main():
    print("=" * 70)
    print("CROSS-JUDGE QUALITY EVALUATION (Claude Haiku 4.5)")
    print("=" * 70)
    print(f"Source: {DYNAMICS}")
    print(f"Original judge scores: {ORIG_QUALITY}")
    print(f"New judge: {JUDGE_MODEL}")
    print()

    with open(DYNAMICS) as f:
        data = json.load(f)
    with open(ORIG_QUALITY) as f:
        orig_quality = json.load(f)

    task_lookup = {t.task_id: t for t in EXAMPLE_TASKS[:4]}

    total_evals = 0
    for cond_name, cond_data in data["conditions"].items():
        for task_data in cond_data:
            for trial in task_data["trials"]:
                if "response_texts" not in trial:
                    continue
                for round_texts in trial["response_texts"]:
                    total_evals += len(round_texts)
    print(f"Total evaluations: {total_evals:,}")

    evaluator = QualityEvaluator(
        method="llm_judge",
        judge_model=JUDGE_MODEL,
        judge_backend="anthropic"
    )

    eval_count = 0
    error_count = 0
    quality_results = {}

    for cond_name, cond_data in data["conditions"].items():
        print(f"\n--- {cond_name} ---")
        cond_qualities = []
        for task_data in cond_data:
            task_id = task_data["task_id"]
            task = task_lookup.get(task_id)
            if task is None:
                continue
            for trial_idx, trial in enumerate(task_data["trials"]):
                if "response_texts" not in trial:
                    continue
                trial_qualities = []
                for round_texts in trial["response_texts"]:
                    round_qualities = []
                    for resp in round_texts:
                        try:
                            score = evaluator.evaluate(resp["text"], task)
                            round_qualities.append(score)
                            eval_count += 1
                        except Exception as e:
                            round_qualities.append(None)
                            error_count += 1
                    trial_qualities.append(round_qualities)
                cond_qualities.append({
                    "task_id": task_id,
                    "trial_idx": trial_idx,
                    "round_qualities": trial_qualities,
                })
                valid = [s for rq in trial_qualities for s in rq if s is not None]
                mean_q = np.mean(valid) if valid else 0
                print(f"  [{task_id}] Trial {trial_idx+1}: q={mean_q:.3f} ({eval_count}/{total_evals})", flush=True)
        quality_results[cond_name] = cond_qualities

    out = {
        "source_file": DYNAMICS,
        "original_judge_file": ORIG_QUALITY,
        "timestamp": datetime.now().isoformat(),
        "evaluator": {
            "method": "llm_judge",
            "model": JUDGE_MODEL,
            "backend": "anthropic",
            "criteria": "coherence, relevance, depth (no originality)",
        },
        "stats": {"total_evaluations": eval_count, "errors": error_count},
        "quality_results": quality_results,
    }
    output_dir = Path("results/quality_cross_judge")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"quality_cross_judge_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {output_file}")

    # ---- Inter-judge agreement ----
    print("\n" + "=" * 70)
    print("INTER-JUDGE AGREEMENT (per-response Pearson r)")
    print("=" * 70)
    from scipy import stats as scipy_stats

    orig_scores = []
    new_scores = []
    for cond_name in quality_results:
        if cond_name not in orig_quality["quality_results"]:
            continue
        new_entries = quality_results[cond_name]
        orig_entries = orig_quality["quality_results"][cond_name]
        for n_e, o_e in zip(new_entries, orig_entries):
            for n_r, o_r in zip(n_e["round_qualities"], o_e["round_qualities"]):
                for n_s, o_s in zip(n_r, o_r):
                    if n_s is not None and o_s is not None:
                        orig_scores.append(o_s)
                        new_scores.append(n_s)
    if len(orig_scores) > 5:
        r, p = scipy_stats.pearsonr(orig_scores, new_scores)
        rho, p_rho = scipy_stats.spearmanr(orig_scores, new_scores)
        print(f"n={len(orig_scores)}")
        print(f"Pearson r = {r:.3f} (p={p:.4e})")
        print(f"Spearman ρ = {rho:.3f} (p={p_rho:.4e})")
        print(f"Mean orig (GPT-4o-mini) = {np.mean(orig_scores):.4f}")
        print(f"Mean new  (Claude H4.5) = {np.mean(new_scores):.4f}")

    # ---- Per-condition: re-run pairwise quality comparisons ----
    print("\n" + "=" * 70)
    print("RE-RUN: DDS alpha=0.5 vs others (under Claude judge)")
    print("=" * 70)
    cond_means = {}
    for cond_name, entries in quality_results.items():
        trial_means = []
        for e in entries:
            scores = [s for rq in e["round_qualities"] for s in rq if s is not None]
            if scores:
                trial_means.append(np.mean(scores))
        cond_means[cond_name] = trial_means
        if trial_means:
            print(f"  {cond_name}: mean={np.mean(trial_means):.4f}, "
                  f"sd={np.std(trial_means, ddof=1):.4f} (n={len(trial_means)})")

    if "dds_alpha_0.5" in cond_means and len(cond_means["dds_alpha_0.5"]) > 0:
        ref = cond_means["dds_alpha_0.5"]
        print()
        for cond_name, vals in cond_means.items():
            if cond_name == "dds_alpha_0.5" or len(vals) == 0:
                continue
            n_pair = min(len(ref), len(vals))
            t, p = scipy_stats.ttest_rel(ref[:n_pair], vals[:n_pair])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  dds_alpha_0.5 vs {cond_name}: t={t:.3f}, p={p:.4f} ({sig})")


if __name__ == "__main__":
    main()

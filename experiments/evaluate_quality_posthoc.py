#!/usr/bin/env python3
"""
Post-hoc Quality Evaluation using LLM-as-Judge

Reads results from dynamics_mapelites experiment (which saves response texts)
and evaluates quality using GPT-4o-mini as judge.

Evaluation criteria: coherence, relevance, depth (NOT originality).
This avoids double-rewarding diversity.

Usage:
    python evaluate_quality_posthoc.py <results_file.json>
    python evaluate_quality_posthoc.py <results_file.json> --yes
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_agent import QualityEvaluator, Task, EXAMPLE_TASKS


def main():
    if len(sys.argv) < 2:
        # Auto-detect latest results file
        results_dir = Path("results/dynamics_mapelites")
        if results_dir.exists():
            files = sorted(results_dir.glob("dynamics_mapelites_*.json"))
            if files:
                input_file = files[-1]
                print(f"Auto-detected results file: {input_file}")
            else:
                print("No results files found in results/dynamics_mapelites/")
                return
        else:
            print("Usage: python evaluate_quality_posthoc.py <results_file.json>")
            return
    else:
        input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"File not found: {input_file}")
        return

    # Load results
    with open(input_file) as f:
        data = json.load(f)

    print("=" * 70)
    print("POST-HOC QUALITY EVALUATION (LLM-as-Judge)")
    print("=" * 70)
    print(f"Input: {input_file}")
    print(f"Start time: {datetime.now().isoformat()}")

    # Build task lookup
    task_lookup = {t.task_id: t for t in EXAMPLE_TASKS[:4]}

    # Count total evaluations needed
    total_evals = 0
    for cond_name, cond_data in data["conditions"].items():
        for task_data in cond_data:
            for trial in task_data["trials"]:
                if "response_texts" not in trial:
                    continue
                for round_texts in trial["response_texts"]:
                    total_evals += len(round_texts)

    print(f"\nTotal evaluations needed: {total_evals:,}")
    print(f"Estimated cost: ~${total_evals * 0.0002:.2f} USD")
    print(f"Estimated time: ~{total_evals * 1.5 / 60:.0f} minutes")
    print()

    if "--yes" not in sys.argv:
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Initialize evaluator
    evaluator = QualityEvaluator(
        method="llm_judge",
        judge_model="gpt-4o-mini",
        judge_backend="openai"
    )

    # Evaluate all responses
    eval_count = 0
    error_count = 0
    quality_results = {}

    for cond_name, cond_data in data["conditions"].items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name}")
        print(f"{'='*60}")

        cond_qualities = []

        for task_data in cond_data:
            task_id = task_data["task_id"]
            task = task_lookup.get(task_id)
            if task is None:
                print(f"  WARNING: Unknown task {task_id}, skipping")
                continue

            for trial_idx, trial in enumerate(task_data["trials"]):
                if "response_texts" not in trial:
                    print(f"  [{task_id}] Trial {trial_idx+1}: no response texts, skipping")
                    continue

                trial_qualities = []
                for round_idx, round_texts in enumerate(trial["response_texts"]):
                    round_qualities = []
                    for resp in round_texts:
                        try:
                            score = evaluator.evaluate(resp["text"], task)
                            round_qualities.append(score)
                            eval_count += 1
                        except Exception as e:
                            print(f"    ERROR evaluating agent {resp['agent_id']}: {e}")
                            round_qualities.append(None)
                            error_count += 1

                    trial_qualities.append(round_qualities)

                cond_qualities.append({
                    "task_id": task_id,
                    "trial_idx": trial_idx,
                    "round_qualities": trial_qualities,
                })

                # Print progress
                valid_scores = [s for rq in trial_qualities for s in rq if s is not None]
                mean_q = np.mean(valid_scores) if valid_scores else 0
                print(f"  [{task_id}] Trial {trial_idx+1}: mean_q={mean_q:.3f} "
                      f"({eval_count}/{total_evals} done)")

        quality_results[cond_name] = cond_qualities

    # Save quality evaluation results
    output = {
        "source_file": str(input_file),
        "timestamp": datetime.now().isoformat(),
        "evaluator": {
            "method": "llm_judge",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "criteria": "coherence, relevance, depth (no originality)",
        },
        "stats": {
            "total_evaluations": eval_count,
            "errors": error_count,
        },
        "quality_results": quality_results,
    }

    output_dir = Path("results/quality_posthoc")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"quality_posthoc_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Quality results saved to: {output_file}")
    print(f"{'='*70}")

    # Print summary
    print(f"\n{'='*70}")
    print("QUALITY SUMMARY")
    print(f"{'='*70}\n")

    from scipy import stats as scipy_stats

    condition_means = {}
    for cond_name, cond_qualities in quality_results.items():
        # Aggregate: mean quality per trial (across all rounds and agents)
        trial_means = []
        # Also track per-round means
        round_means_all = {}

        for entry in cond_qualities:
            all_scores = []
            for round_idx, rq in enumerate(entry["round_qualities"]):
                valid = [s for s in rq if s is not None]
                if valid:
                    round_mean = np.mean(valid)
                    all_scores.extend(valid)
                    if round_idx not in round_means_all:
                        round_means_all[round_idx] = []
                    round_means_all[round_idx].append(round_mean)

            if all_scores:
                trial_means.append(np.mean(all_scores))

        if trial_means:
            mean_q = np.mean(trial_means)
            std_q = np.std(trial_means, ddof=1)
            condition_means[cond_name] = trial_means
            print(f"{cond_name}:")
            print(f"  Overall quality: {mean_q:.4f} +/- {std_q:.4f} (n={len(trial_means)})")
            for r_idx in sorted(round_means_all.keys()):
                rm = round_means_all[r_idx]
                print(f"  Round {r_idx}: {np.mean(rm):.4f} +/- {np.std(rm, ddof=1):.4f}")
            print()

    # Pairwise comparisons
    if "dds_alpha_0.5" in condition_means:
        print("Pairwise comparisons (DDS alpha=0.5 vs others, quality):")
        print("-" * 60)
        ref = condition_means["dds_alpha_0.5"]

        for cond_name, cond_vals in condition_means.items():
            if cond_name == "dds_alpha_0.5":
                continue
            n_paired = min(len(ref), len(cond_vals))
            if n_paired < 3:
                continue
            t_stat, p_val = scipy_stats.ttest_rel(ref[:n_paired], cond_vals[:n_paired])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            print(f"  dds_alpha_0.5 vs {cond_name}: t={t_stat:.3f}, p={p_val:.4f} ({sig})")

    print(f"\nEvaluation completed at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Alpha Sweep with Extended Alpha Values for Publication Figure.
Runs: α = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0] with n=20 trials each.
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.extended_experiment import ExtendedExperimentRunner, ExtendedExperimentConfig
from src.llm_agent import EXAMPLE_TASKS
from tqdm import tqdm
import json
import numpy as np

def main():
    print("=" * 70)
    print("ALPHA SWEEP EXTENDED: 6 ALPHA VALUES FOR PUBLICATION")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print()

    # Configuration
    config = ExtendedExperimentConfig(
        name="alpha_sweep_extended",
        n_agents=6,
        n_rounds=3,
        n_trials=1,
        backend="openai",
        model="gpt-4o-mini",
        embedding_model="all-MiniLM-L6-v2",
        output_dir="results/alpha_sweep_extended",
        seed=42,
        dds_alpha_values=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        dds_beta=2.0,
        dds_bandwidth=0.3,
    )

    n_trials_per_task = 20
    n_tasks = 4

    print("Configuration:")
    print(f"  Backend: {config.backend}")
    print(f"  Model: {config.model}")
    print(f"  Agents: {config.n_agents}")
    print(f"  Rounds: {config.n_rounds}")
    print(f"  Trials per task per alpha: {n_trials_per_task}")
    print(f"  Tasks: {n_tasks}")
    print(f"  Alpha values: {config.dds_alpha_values}")
    print()

    # Cost estimation
    api_calls_per_trial = config.n_agents * config.n_rounds
    total_trials = n_trials_per_task * len(config.dds_alpha_values) * n_tasks
    total_api_calls = total_trials * api_calls_per_trial

    estimated_cost = total_api_calls * 0.00015 / 1000 * 150
    estimated_minutes = total_api_calls * 2 / 60

    print(f"Estimated:")
    print(f"  Total trials: {total_trials}")
    print(f"  Total API calls: {total_api_calls:,}")
    print(f"  Cost: ~${estimated_cost:.2f} USD")
    print(f"  Time: ~{estimated_minutes:.0f} minutes ({estimated_minutes/60:.1f} hours)")
    print()

    response = input("Continue with experiment? [y/N]: ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return None

    # Create runner
    runner = ExtendedExperimentRunner(config)

    print("\nStarting alpha sweep experiment...")
    print("-" * 70)

    try:
        all_results = {
            "config": config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "n_trials_per_task": n_trials_per_task,
            "experiments": {}
        }

        tasks = EXAMPLE_TASKS[:n_tasks]
        all_alpha_sweep_data = []

        for task_idx, task in enumerate(tasks):
            print(f"\n[Task {task_idx+1}/{len(tasks)}] {task.task_id}")

            task_data = {
                "task_id": task.task_id,
                "trials": []
            }

            # Run n_trials for this task
            for trial_idx in tqdm(range(n_trials_per_task), desc=f"  Trials"):
                # Run single alpha sweep
                result = runner.run_alpha_sweep_comparison(task)

                # Extract alpha sweep data
                sweep = result.get("alpha_sweep", {})
                trial_data = {
                    "trial": trial_idx
                }

                # Store all alpha values
                for alpha in config.dds_alpha_values:
                    alpha_str = str(float(alpha))
                    if alpha_str in sweep:
                        trial_data[f"alpha_{alpha}"] = sweep[alpha_str]

                task_data["trials"].append(trial_data)

            all_alpha_sweep_data.append(task_data)

            # Show intermediate statistics for each alpha
            print(f"\n  Task {task.task_id} results:")
            for alpha in config.dds_alpha_values:
                alpha_key = f"alpha_{alpha}"
                divs = [t[alpha_key]["diversity"] for t in task_data["trials"] if alpha_key in t]
                if divs:
                    print(f"    α={alpha:3.1f}: {np.mean(divs):.4f} ± {np.std(divs):.4f}")

        all_results["experiments"]["alpha_sweep"] = all_alpha_sweep_data

        # Calculate summary statistics
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)

        # Flatten data for analysis
        all_alpha_data = {alpha: [] for alpha in config.dds_alpha_values}

        for task_data in all_alpha_sweep_data:
            for trial_data in task_data["trials"]:
                for alpha in config.dds_alpha_values:
                    alpha_key = f"alpha_{alpha}"
                    if alpha_key in trial_data:
                        all_alpha_data[alpha].append(trial_data[alpha_key]["diversity"])

        print(f"\nOverall Statistics (n={n_trials_per_task * n_tasks} per alpha):")
        for alpha in config.dds_alpha_values:
            divs = all_alpha_data[alpha]
            if divs:
                print(f"  α={alpha:3.1f}: {np.mean(divs):.4f} ± {np.std(divs):.4f} (n={len(divs)})")

        # Statistical tests (comparing adjacent alpha values)
        print(f"\nPairwise Statistical Tests:")
        from scipy import stats as scipy_stats

        for i in range(len(config.dds_alpha_values) - 1):
            alpha1 = config.dds_alpha_values[i]
            alpha2 = config.dds_alpha_values[i + 1]

            divs1 = all_alpha_data[alpha1]
            divs2 = all_alpha_data[alpha2]

            if divs1 and divs2:
                # For paired comparison (same task, same trial, different alpha)
                # We need to pair them by task and trial
                paired_divs1 = []
                paired_divs2 = []

                for task_data in all_alpha_sweep_data:
                    for trial_data in task_data["trials"]:
                        key1 = f"alpha_{alpha1}"
                        key2 = f"alpha_{alpha2}"
                        if key1 in trial_data and key2 in trial_data:
                            paired_divs1.append(trial_data[key1]["diversity"])
                            paired_divs2.append(trial_data[key2]["diversity"])

                if paired_divs1:
                    t_stat, p_value = scipy_stats.ttest_rel(paired_divs1, paired_divs2)
                    mean_diff = np.mean(np.array(paired_divs1) - np.array(paired_divs2))

                    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"  α={alpha1:3.1f} vs α={alpha2:3.1f}: Δ={mean_diff:+.4f}, p={p_value:.4f} {sig_marker}")

        # Save summary
        all_results["summary"] = {
            "n_trials_per_task": n_trials_per_task,
            "total_observations_per_alpha": len(divs1),
            "alpha_statistics": {}
        }

        for alpha in config.dds_alpha_values:
            divs = all_alpha_data[alpha]
            if divs:
                all_results["summary"]["alpha_statistics"][str(alpha)] = {
                    "mean": float(np.mean(divs)),
                    "std": float(np.std(divs)),
                    "sem": float(scipy_stats.sem(divs)),
                    "n": len(divs)
                }

        # Save results
        os.makedirs(config.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(config.output_dir, f"alpha_sweep_extended_{timestamp}.json")

        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=2)

        print()
        print("=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        print(f"End time: {datetime.now().isoformat()}")
        print(f"Results saved: {filepath}")
        print()
        print("Next steps:")
        print("  1. Run: python generate_figure4_from_extended.py")
        print("  2. Check: paper/figures/fig4_phase_transition.pdf")
        print()

        return all_results

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        print("Partial results may be incomplete.")
        return None
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()

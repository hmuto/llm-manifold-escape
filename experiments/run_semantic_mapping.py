#!/usr/bin/env python3
"""
Semantic Mapping Experiment with Real API Data.
Runs n_agents=8, n_rounds=3 for richer semantic visualization.
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.extended_experiment import ExtendedExperimentRunner, ExtendedExperimentConfig
from src.llm_agent import EXAMPLE_TASKS
import json

def main():
    print("=" * 70)
    print("SEMANTIC MAPPING EXPERIMENT - REAL API DATA")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print()

    # Configuration for semantic mapping with 8 agents
    config = ExtendedExperimentConfig(
        name="semantic_mapping_n8",
        n_agents=8,
        n_rounds=3,
        n_trials=1,
        backend="openai",
        model="gpt-4o-mini",
        embedding_model="all-MiniLM-L6-v2",
        output_dir="results/semantic_mapping",
        seed=42,
        dds_alpha_values=[0.0],  # Only need α=0.0 for maximum diversity
        dds_beta=2.0,
        dds_bandwidth=0.3,
        map_n_bins=5
    )

    print("Configuration:")
    print(f"  Backend: {config.backend}")
    print(f"  Model: {config.model}")
    print(f"  Agents: {config.n_agents}")
    print(f"  Rounds: {config.n_rounds}")
    print(f"  Task: creative_1 (semantic mapping demonstration)")
    print()

    # Cost estimation
    api_calls = config.n_agents * config.n_rounds
    estimated_cost = api_calls * 0.000165  # Per call estimate
    estimated_time = api_calls * 2 / 60  # seconds to minutes

    print(f"Estimated:")
    print(f"  API calls: {api_calls}")
    print(f"  Cost: ~${estimated_cost:.4f} USD")
    print(f"  Time: ~{estimated_time:.1f} minutes")
    print()

    response = input("Continue with experiment? [y/N]: ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return None

    # Create runner
    runner = ExtendedExperimentRunner(config)

    print("\nStarting semantic mapping experiment...")
    print("-" * 70)

    try:
        # Use creative_1 task for semantic mapping
        task = EXAMPLE_TASKS[0]  # creative_1

        print(f"\nTask: {task.task_id}")
        print()

        print("Running semantic interpretation with α=0.0 (maximum diversity)...")
        result = runner.run_semantic_interpretation(task, alpha=0.0)

        # Save results
        os.makedirs(config.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(config.output_dir, f"semantic_mapping_{timestamp}.json")

        output = {
            "config": config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "task_id": task.task_id,
            "experiment": result
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        # Print semantic interpretation results
        if 'semantic_interpretation' in result:
            sem = result['semantic_interpretation']
            n_responses = len(sem['coordinates'])
            axes = sem['axes']
            print(f"\nSemantic Mapping Generated:")
            print(f"  Responses: {n_responses}")
            print(f"  PC1: {axes[0]['label']} ({axes[0]['explained_variance']*100:.1f}%)")
            print(f"  PC2: {axes[1]['label']} ({axes[1]['explained_variance']*100:.1f}%)")
            total_var = sum(a['explained_variance'] for a in axes)
            print(f"  Total variance: {total_var*100:.1f}%")

        print()
        print("=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        print(f"End time: {datetime.now().isoformat()}")
        print(f"Results saved: {filepath}")
        print()
        print("Next steps:")
        print("  1. Update generate_final_figures.py to use this data")
        print("  2. Regenerate Figure 3")
        print()

        return output

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        return None
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()

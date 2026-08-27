#!/usr/bin/env python3
"""
Extended Rounds Experiment: Do DDS/MAP-Elites continue expanding diversity?

Tests whether cumulative exploration continues beyond 3 rounds or saturates.

3 Conditions:
1. DDS alpha=0.5 (optimal from alpha sweep)
2. MAP-Elites (archive-based QD baseline)
3. Independent (repeated independent generation, no selection)

Settings: N=8, 7 rounds, 5 trials, 4 tasks, GPT-4o-mini
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_agent import MultiAgentSystem, AgentConfig, Task, EXAMPLE_TASKS
from src.protocols import (
    DDSProtocol, DDSConfig,
    IndependentProtocol, ProtocolConfig
)
from src.density_selection import AgentResponse
from src.map_elites import MAPElitesSelector, MAPElitesConfig


def compute_diversity(responses):
    """Compute mean pairwise cosine distance from Response objects."""
    embeddings = np.array([r.embedding for r in responses])
    if len(embeddings) < 2:
        return 0.0
    from scipy.spatial.distance import cdist
    dists = cdist(embeddings, embeddings, metric='cosine')
    n = len(embeddings)
    upper = dists[np.triu_indices(n, k=1)]
    return float(np.mean(upper))


def extract_texts(responses):
    """Extract response texts from Response objects."""
    return [{"agent_id": r.agent_id, "text": r.text} for r in responses]


def run_dds_condition(system, task, alpha, n_rounds, n_survive):
    """Run DDS protocol for extended rounds."""
    config = DDSConfig(
        n_rounds=n_rounds,
        n_agents=system.n_agents,
        alpha=alpha,
        beta=2.0,
        bandwidth=0.3,
        n_survive=n_survive
    )
    protocol = DDSProtocol(config)
    result = protocol.run(system, task, quality_evaluator=None)

    round_diversities = []
    response_texts = []
    for round_responses in result["round_history"]:
        round_diversities.append(compute_diversity(round_responses))
        response_texts.append(extract_texts(round_responses))

    return {
        "round_diversities": round_diversities,
        "final_diversity": round_diversities[-1],
        "n_rounds": len(round_diversities),
        "response_texts": response_texts,
    }


def run_independent_multi_round(system, task, n_rounds):
    """Run Independent generation for n_rounds (no selection between rounds)."""
    config = ProtocolConfig(n_rounds=1, n_agents=system.n_agents)
    protocol = IndependentProtocol(config)

    round_diversities = []
    response_texts = []

    for round_idx in range(n_rounds):
        result = protocol.run(system, task, quality_evaluator=None)
        responses = result["final_responses"]
        round_diversities.append(compute_diversity(responses))
        response_texts.append(extract_texts(responses))

    return {
        "round_diversities": round_diversities,
        "final_diversity": round_diversities[-1],
        "n_rounds": len(round_diversities),
        "response_texts": response_texts,
    }


def run_mapelites_condition(system, task, n_rounds):
    """Run MAP-Elites for extended rounds."""
    me_config = MAPElitesConfig(n_bins_per_dim=5, n_behavior_dims=2)
    selector = MAPElitesSelector(me_config, behavior_method="pca")

    system.reset_all()

    round_diversities = []
    response_texts = []
    archive_coverages = []

    for round_idx in range(n_rounds):
        if round_idx == 0:
            responses = system.generate_responses(task)
        else:
            elites = selector.archive.get_all_elites()

            def context_provider(agent_id, _elites=elites):
                elite_texts = [
                    f"Selected response: {e.response_text}"
                    for e in _elites
                ]
                return "\n\n".join(elite_texts)

            me_task = Task(
                task_id=f"{task.task_id}_round{round_idx}",
                prompt=f"{task.prompt}\n\nBuild upon or differentiate from the context.",
                category=task.category
            )
            responses = system.generate_responses(me_task, context_provider)

        round_diversities.append(compute_diversity(responses))
        response_texts.append(extract_texts(responses))

        agent_responses = [
            AgentResponse(
                agent_id=r.agent_id,
                response_text=r.text,
                embedding=r.embedding,
                quality_score=0.75,
                generation=round_idx
            )
            for r in responses
        ]
        stats = selector.update_archive(agent_responses)
        archive_coverages.append(stats["coverage"])

    return {
        "round_diversities": round_diversities,
        "final_diversity": round_diversities[-1],
        "n_rounds": len(round_diversities),
        "response_texts": response_texts,
        "archive_coverage": archive_coverages,
    }


def main():
    print("=" * 70)
    print("EXTENDED ROUNDS EXPERIMENT")
    print("Do DDS/MAP-Elites continue expanding diversity beyond 3 rounds?")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print()

    # Configuration
    N_AGENTS = 8
    N_ROUNDS = 7
    N_SURVIVE = 5
    N_TRIALS = 5
    BACKEND = "openai"
    MODEL = "gpt-4o-mini"

    tasks = EXAMPLE_TASKS[:4]

    print("Configuration:")
    print(f"  Backend: {BACKEND}")
    print(f"  Model: {MODEL}")
    print(f"  N_agents: {N_AGENTS}")
    print(f"  N_rounds: {N_ROUNDS}")
    print(f"  N_survive (DDS): {N_SURVIVE}")
    print(f"  N_trials per task: {N_TRIALS}")
    print(f"  Tasks: {[t.task_id for t in tasks]}")
    print(f"  Conditions: DDS alpha=0.5, MAP-Elites, Independent")
    print()

    # Cost estimation
    n_tasks = len(tasks)
    dds_calls = n_tasks * N_TRIALS * N_AGENTS * N_ROUNDS
    me_calls = n_tasks * N_TRIALS * N_AGENTS * N_ROUNDS
    indep_calls = n_tasks * N_TRIALS * N_AGENTS * N_ROUNDS
    total_calls = dds_calls + me_calls + indep_calls

    print(f"Estimated API calls: {total_calls:,}")
    print(f"  DDS alpha=0.5: {dds_calls:,}")
    print(f"  MAP-Elites: {me_calls:,}")
    print(f"  Independent: {indep_calls:,}")
    print(f"Estimated cost: ~${total_calls * 0.0002:.2f} USD")
    print(f"Estimated time: ~{total_calls * 2 / 60:.0f} minutes")
    print()

    if "--yes" not in sys.argv:
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Create agent system
    agent_config = AgentConfig(
        agent_id=0,
        backend=BACKEND,
        model=MODEL,
        temperature=0.7,
        max_tokens=512
    )
    system = MultiAgentSystem(
        n_agents=N_AGENTS,
        agent_config_template=agent_config,
        embedding_model="all-MiniLM-L6-v2"
    )

    # Results storage
    all_results = {
        "config": {
            "n_agents": N_AGENTS,
            "n_rounds": N_ROUNDS,
            "n_survive": N_SURVIVE,
            "n_trials": N_TRIALS,
            "backend": BACKEND,
            "model": MODEL,
            "tasks": [t.task_id for t in tasks],
        },
        "timestamp": datetime.now().isoformat(),
        "conditions": {}
    }

    output_dir = Path("results/extended_rounds")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Run DDS alpha=0.5
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Condition: DDS alpha=0.5")
    print(f"{'='*60}")

    dds_results = []
    for task_idx, task in enumerate(tasks):
        task_trials = []
        for trial in range(N_TRIALS):
            print(f"  [{task.task_id}] Trial {trial+1}/{N_TRIALS}...", end="", flush=True)
            try:
                result = run_dds_condition(system, task, 0.5, N_ROUNDS, N_SURVIVE)
                task_trials.append(result)
                divs = result['round_diversities']
                print(f" R0={divs[0]:.3f} R3={divs[3]:.3f} R6={divs[6]:.3f}")
            except Exception as e:
                print(f" ERROR: {e}")
                task_trials.append(None)

        dds_results.append({
            "task_id": task.task_id,
            "trials": [t for t in task_trials if t is not None]
        })

    all_results["conditions"]["dds_alpha_0.5"] = dds_results

    # Save intermediate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"extended_rounds_{timestamp}_partial.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("  [Intermediate save complete]")

    # ============================================================
    # Run MAP-Elites
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Condition: MAP-Elites")
    print(f"{'='*60}")

    me_results = []
    for task_idx, task in enumerate(tasks):
        task_trials = []
        for trial in range(N_TRIALS):
            print(f"  [{task.task_id}] Trial {trial+1}/{N_TRIALS}...", end="", flush=True)
            try:
                result = run_mapelites_condition(system, task, N_ROUNDS)
                task_trials.append(result)
                divs = result['round_diversities']
                cov = result['archive_coverage'][-1]
                print(f" R0={divs[0]:.3f} R6={divs[6]:.3f} cov={cov:.1%}")
            except Exception as e:
                print(f" ERROR: {e}")
                task_trials.append(None)

        me_results.append({
            "task_id": task.task_id,
            "trials": [t for t in task_trials if t is not None]
        })

    all_results["conditions"]["map_elites"] = me_results

    # Save intermediate
    with open(output_dir / f"extended_rounds_{timestamp}_partial.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("  [Intermediate save complete]")

    # ============================================================
    # Run Independent (multi-round, no selection)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Condition: Independent (7 independent rounds)")
    print(f"{'='*60}")

    indep_results = []
    for task_idx, task in enumerate(tasks):
        task_trials = []
        for trial in range(N_TRIALS):
            print(f"  [{task.task_id}] Trial {trial+1}/{N_TRIALS}...", end="", flush=True)
            try:
                result = run_independent_multi_round(system, task, N_ROUNDS)
                task_trials.append(result)
                divs = result['round_diversities']
                print(f" R0={divs[0]:.3f} R3={divs[3]:.3f} R6={divs[6]:.3f}")
            except Exception as e:
                print(f" ERROR: {e}")
                task_trials.append(None)

        indep_results.append({
            "task_id": task.task_id,
            "trials": [t for t in task_trials if t is not None]
        })

    all_results["conditions"]["independent"] = indep_results

    # ============================================================
    # Save final results
    # ============================================================
    output_file = output_dir / f"extended_rounds_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    # ============================================================
    # Summary statistics
    # ============================================================
    from scipy import stats as scipy_stats

    print(f"\n{'='*70}")
    print("SUMMARY: PER-ROUND DIVERSITY")
    print(f"{'='*70}\n")

    for cond_name, cond_data in all_results["conditions"].items():
        round_divs_by_round = {}
        for task_data in cond_data:
            for trial in task_data["trials"]:
                for r_idx, rd in enumerate(trial["round_diversities"]):
                    if r_idx not in round_divs_by_round:
                        round_divs_by_round[r_idx] = []
                    round_divs_by_round[r_idx].append(rd)

        print(f"{cond_name}:")
        for r_idx in sorted(round_divs_by_round.keys()):
            rd = round_divs_by_round[r_idx]
            print(f"  Round {r_idx}: {np.mean(rd):.4f} +/- {np.std(rd, ddof=1):.4f} (n={len(rd)})")
        print()

    print(f"\nExperiment completed at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

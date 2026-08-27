#!/usr/bin/env python3
"""
Dynamics Experiment with MAP-Elites: Process vs Archive Comparison

Extends rigorous_dynamics experiment by adding:
1. MAP-Elites condition for direct "Process vs Archive" comparison
2. Response text saving for post-hoc quality evaluation

6 Conditions:
1. DDS alpha=0.0 (no density penalty - baseline)
2. DDS alpha=0.5 (optimal from alpha sweep)
3. DDS alpha=1.0 (standard density penalty)
4. Debate (diversity collapse baseline)
5. Independent (single-round, no interaction)
6. MAP-Elites (archive-based QD baseline)

Settings: N=8, 3 rounds, 5 trials, 4 tasks, GPT-4o-mini
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_agent import MultiAgentSystem, AgentConfig, Task, EXAMPLE_TASKS
from src.protocols import (
    DDSProtocol, DDSConfig,
    DebateProtocol, DebateConfig,
    IndependentProtocol, ProtocolConfig
)
from src.density_selection import (
    DensityDependentSelector, SelectionConfig, AgentResponse
)
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


def count_niches(responses, eps=0.5):
    """Count niches using DBSCAN."""
    embeddings = np.array([r.embedding for r in responses])
    if len(embeddings) < 2:
        return 1
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine')
    labels = clustering.fit_predict(embeddings)
    n_clusters = len(set(labels) - {-1})
    return max(1, n_clusters)


def extract_texts(responses):
    """Extract response texts from Response objects."""
    return [{"agent_id": r.agent_id, "text": r.text} for r in responses]


def run_dds_condition(system, task, alpha, n_rounds, n_survive):
    """Run DDS protocol and track per-round diversity + texts."""
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
    round_niches = []
    response_texts = []
    for round_responses in result["round_history"]:
        round_diversities.append(compute_diversity(round_responses))
        round_niches.append(count_niches(round_responses))
        response_texts.append(extract_texts(round_responses))

    return {
        "round_diversities": round_diversities,
        "round_niches": round_niches,
        "final_diversity": round_diversities[-1],
        "final_niches": round_niches[-1],
        "n_rounds": len(round_diversities),
        "response_texts": response_texts,
    }


def run_debate_condition(system, task, n_rounds):
    """Run Debate protocol and track per-round diversity + texts."""
    config = DebateConfig(
        n_rounds=n_rounds,
        n_agents=system.n_agents,
        show_all_responses=True
    )
    protocol = DebateProtocol(config)
    result = protocol.run(system, task, quality_evaluator=None)

    round_diversities = []
    round_niches = []
    response_texts = []
    for round_responses in result["round_history"]:
        round_diversities.append(compute_diversity(round_responses))
        round_niches.append(count_niches(round_responses))
        response_texts.append(extract_texts(round_responses))

    return {
        "round_diversities": round_diversities,
        "round_niches": round_niches,
        "final_diversity": round_diversities[-1],
        "final_niches": round_niches[-1],
        "n_rounds": len(round_diversities),
        "response_texts": response_texts,
    }


def run_independent_condition(system, task):
    """Run Independent protocol (single round, no interaction) + texts."""
    config = ProtocolConfig(n_rounds=1, n_agents=system.n_agents)
    protocol = IndependentProtocol(config)
    result = protocol.run(system, task, quality_evaluator=None)

    responses = result["final_responses"]
    div = compute_diversity(responses)
    n = count_niches(responses)

    return {
        "round_diversities": [div],
        "round_niches": [n],
        "final_diversity": div,
        "final_niches": n,
        "n_rounds": 1,
        "response_texts": [extract_texts(responses)],
    }


def run_mapelites_condition(system, task, n_rounds):
    """
    Run MAP-Elites protocol and track per-round metrics.

    Multi-round MAP-Elites:
    - Round 0: Generate responses -> fit PCA -> add to 5x5 grid archive
    - Round 1+: Present archive elites as context -> regenerate -> update archive
    - Track: response diversity, archive coverage, archive elite diversity
    """
    me_config = MAPElitesConfig(n_bins_per_dim=5, n_behavior_dims=2)
    selector = MAPElitesSelector(me_config, behavior_method="pca")

    system.reset_all()

    round_diversities = []
    round_niches = []
    response_texts = []
    archive_coverages = []
    archive_diversities = []

    for round_idx in range(n_rounds):
        if round_idx == 0:
            # Initial generation (no context)
            responses = system.generate_responses(task)
        else:
            # Get archive elites as context
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

        # Compute metrics on generated responses
        div = compute_diversity(responses)
        n = count_niches(responses)
        round_diversities.append(div)
        round_niches.append(n)

        # Save texts
        response_texts.append(extract_texts(responses))

        # Convert to AgentResponse and update archive
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

        # Compute diversity among archive elites
        archive_elites = selector.archive.get_all_elites()
        if len(archive_elites) >= 2:
            a_div = compute_diversity(archive_elites)
        else:
            a_div = 0.0
        archive_diversities.append(a_div)

    return {
        "round_diversities": round_diversities,
        "round_niches": round_niches,
        "final_diversity": round_diversities[-1],
        "final_niches": round_niches[-1],
        "n_rounds": len(round_diversities),
        "response_texts": response_texts,
        "archive_coverage": archive_coverages,
        "archive_diversity": archive_diversities,
    }


def main():
    print("=" * 70)
    print("DYNAMICS EXPERIMENT WITH MAP-ELITES")
    print("Process vs Archive Comparison")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print()

    # Configuration
    N_AGENTS = 8
    N_ROUNDS = 3
    N_SURVIVE = 5
    N_TRIALS = 5
    BACKEND = "openai"
    MODEL = "gpt-4o-mini"
    ALPHA_VALUES = [0.0, 0.5, 1.0]

    tasks = EXAMPLE_TASKS[:4]

    print("Configuration:")
    print(f"  Backend: {BACKEND}")
    print(f"  Model: {MODEL}")
    print(f"  N_agents: {N_AGENTS}")
    print(f"  N_rounds: {N_ROUNDS}")
    print(f"  N_survive (DDS): {N_SURVIVE}")
    print(f"  N_trials per task: {N_TRIALS}")
    print(f"  Tasks: {[t.task_id for t in tasks]}")
    print(f"  Alpha values: {ALPHA_VALUES}")
    print()

    # Cost estimation
    dds_calls = len(ALPHA_VALUES) * len(tasks) * N_TRIALS * N_AGENTS * N_ROUNDS
    debate_calls = len(tasks) * N_TRIALS * N_AGENTS * N_ROUNDS
    indep_calls = len(tasks) * N_TRIALS * N_AGENTS * 1
    me_calls = len(tasks) * N_TRIALS * N_AGENTS * N_ROUNDS
    total_calls = dds_calls + debate_calls + indep_calls + me_calls

    print(f"Estimated API calls: {total_calls:,}")
    print(f"  DDS ({len(ALPHA_VALUES)} alphas): {dds_calls:,}")
    print(f"  Debate: {debate_calls:,}")
    print(f"  Independent: {indep_calls:,}")
    print(f"  MAP-Elites: {me_calls:,}")
    print(f"Estimated cost: ~${total_calls * 0.0002:.2f} USD")
    print(f"Estimated time: ~{total_calls * 2 / 60:.0f} minutes")
    print()

    # Auto-confirm if --yes flag is provided
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
            "alpha_values": ALPHA_VALUES,
            "tasks": [t.task_id for t in tasks],
        },
        "timestamp": datetime.now().isoformat(),
        "conditions": {}
    }

    # ============================================================
    # Run DDS conditions
    # ============================================================
    for alpha in ALPHA_VALUES:
        condition_name = f"dds_alpha_{alpha}"
        print(f"\n{'='*60}")
        print(f"Condition: DDS alpha={alpha}")
        print(f"{'='*60}")

        condition_results = []
        for task_idx, task in enumerate(tasks):
            task_trials = []
            for trial in range(N_TRIALS):
                print(f"  [{task.task_id}] Trial {trial+1}/{N_TRIALS}...", end="", flush=True)
                try:
                    result = run_dds_condition(system, task, alpha, N_ROUNDS, N_SURVIVE)
                    task_trials.append(result)
                    print(f" div={result['final_diversity']:.4f}")
                except Exception as e:
                    print(f" ERROR: {e}")
                    task_trials.append(None)

            condition_results.append({
                "task_id": task.task_id,
                "trials": [t for t in task_trials if t is not None]
            })

        all_results["conditions"][condition_name] = condition_results

    # ============================================================
    # Run Debate condition
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Condition: Debate")
    print(f"{'='*60}")

    debate_results = []
    for task_idx, task in enumerate(tasks):
        task_trials = []
        for trial in range(N_TRIALS):
            print(f"  [{task.task_id}] Trial {trial+1}/{N_TRIALS}...", end="", flush=True)
            try:
                result = run_debate_condition(system, task, N_ROUNDS)
                task_trials.append(result)
                print(f" div={result['final_diversity']:.4f}")
            except Exception as e:
                print(f" ERROR: {e}")
                task_trials.append(None)

        debate_results.append({
            "task_id": task.task_id,
            "trials": [t for t in task_trials if t is not None]
        })

    all_results["conditions"]["debate"] = debate_results

    # ============================================================
    # Run Independent condition
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Condition: Independent")
    print(f"{'='*60}")

    indep_results = []
    for task_idx, task in enumerate(tasks):
        task_trials = []
        for trial in range(N_TRIALS):
            print(f"  [{task.task_id}] Trial {trial+1}/{N_TRIALS}...", end="", flush=True)
            try:
                result = run_independent_condition(system, task)
                task_trials.append(result)
                print(f" div={result['final_diversity']:.4f}")
            except Exception as e:
                print(f" ERROR: {e}")
                task_trials.append(None)

        indep_results.append({
            "task_id": task.task_id,
            "trials": [t for t in task_trials if t is not None]
        })

    all_results["conditions"]["independent"] = indep_results

    # ============================================================
    # Run MAP-Elites condition
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
                cov = result['archive_coverage'][-1]
                print(f" div={result['final_diversity']:.4f}, cov={cov:.2%}")
            except Exception as e:
                print(f" ERROR: {e}")
                task_trials.append(None)

        me_results.append({
            "task_id": task.task_id,
            "trials": [t for t in task_trials if t is not None]
        })

    all_results["conditions"]["map_elites"] = me_results

    # ============================================================
    # Save results
    # ============================================================
    output_dir = Path("results/dynamics_mapelites")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"dynamics_mapelites_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    # ============================================================
    # Print summary statistics
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}\n")

    from scipy import stats as scipy_stats

    for cond_name, cond_data in all_results["conditions"].items():
        all_final_divs = []
        round_divs_by_round = {}

        for task_data in cond_data:
            for trial in task_data["trials"]:
                all_final_divs.append(trial["final_diversity"])
                for r_idx, rd in enumerate(trial["round_diversities"]):
                    if r_idx not in round_divs_by_round:
                        round_divs_by_round[r_idx] = []
                    round_divs_by_round[r_idx].append(rd)

        n = len(all_final_divs)
        if n == 0:
            continue

        print(f"{cond_name}:")
        print(f"  Final diversity: {np.mean(all_final_divs):.4f} +/- {np.std(all_final_divs, ddof=1):.4f} (n={n})")

        for r_idx in sorted(round_divs_by_round.keys()):
            rd = round_divs_by_round[r_idx]
            print(f"  Round {r_idx}: {np.mean(rd):.4f} +/- {np.std(rd, ddof=1):.4f}")

        # MAP-Elites specific stats
        if cond_name == "map_elites":
            all_cov = []
            all_adiv = []
            for task_data in cond_data:
                for trial in task_data["trials"]:
                    all_cov.append(trial["archive_coverage"][-1])
                    all_adiv.append(trial["archive_diversity"][-1])
            if all_cov:
                print(f"  Archive coverage: {np.mean(all_cov):.4f} +/- {np.std(all_cov, ddof=1):.4f}")
                print(f"  Archive diversity: {np.mean(all_adiv):.4f} +/- {np.std(all_adiv, ddof=1):.4f}")
        print()

    # Pairwise comparisons: DDS alpha=0.5 vs others
    print("Pairwise comparisons (paired t-tests on final diversity):")
    print("-" * 60)

    ref_cond = "dds_alpha_0.5"
    ref_divs = []
    for task_data in all_results["conditions"].get(ref_cond, []):
        for trial in task_data["trials"]:
            ref_divs.append(trial["final_diversity"])

    for cond_name in all_results["conditions"]:
        if cond_name == ref_cond:
            continue

        cond_divs = []
        for task_data in all_results["conditions"][cond_name]:
            for trial in task_data["trials"]:
                cond_divs.append(trial["final_diversity"])

        n_paired = min(len(ref_divs), len(cond_divs))
        if n_paired < 3:
            continue

        t_stat, p_val = scipy_stats.ttest_rel(ref_divs[:n_paired], cond_divs[:n_paired])
        diff = np.array(ref_divs[:n_paired]) - np.array(cond_divs[:n_paired])
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"  {ref_cond} vs {cond_name}: t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f} ({sig})")

    print(f"\nExperiment completed at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

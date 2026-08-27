#!/usr/bin/env python3
"""
Robustness Check: Dynamics Experiment with Claude Haiku 4.5

Replicates the core finding (DDS > Debate, DDS > Independent in cumulative
diversity) using a different LLM backend to address the "single model"
limitation for TMLR submission.

Conditions (3, the core comparison):
1. DDS alpha=0.5 (optimal from main experiment)
2. Debate (diversity collapse baseline)
3. Independent (no interaction baseline)

Settings: N=8, 3 rounds, 5 trials, 4 tasks, Claude Haiku 4.5
Total API calls: ~1,120, Time: ~1.5h, Cost: ~$3-5
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_agent import MultiAgentSystem, AgentConfig, EXAMPLE_TASKS
from src.protocols import (
    DDSProtocol, DDSConfig,
    DebateProtocol, DebateConfig,
    IndependentProtocol, ProtocolConfig
)


def compute_diversity(responses):
    embeddings = np.array([r.embedding for r in responses])
    if len(embeddings) < 2:
        return 0.0
    from scipy.spatial.distance import cdist
    dists = cdist(embeddings, embeddings, metric='cosine')
    n = len(embeddings)
    upper = dists[np.triu_indices(n, k=1)]
    return float(np.mean(upper))


def count_niches(responses, eps=0.5):
    embeddings = np.array([r.embedding for r in responses])
    if len(embeddings) < 2:
        return 1
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine')
    labels = clustering.fit_predict(embeddings)
    n_clusters = len(set(labels) - {-1})
    return max(1, n_clusters)


def extract_texts(responses):
    return [{"agent_id": r.agent_id, "text": r.text} for r in responses]


def run_dds_condition(system, task, alpha, n_rounds, n_survive):
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

    round_diversities, round_niches, response_texts = [], [], []
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
    config = DebateConfig(
        n_rounds=n_rounds,
        n_agents=system.n_agents,
        show_all_responses=True
    )
    protocol = DebateProtocol(config)
    result = protocol.run(system, task, quality_evaluator=None)

    round_diversities, round_niches, response_texts = [], [], []
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


def main():
    print("=" * 70)
    print("ROBUSTNESS CHECK: Dynamics with Claude Haiku 4.5")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print()

    N_AGENTS = 8
    N_ROUNDS = 3
    N_SURVIVE = 5
    N_TRIALS = 5
    BACKEND = "anthropic"
    MODEL = "claude-haiku-4-5-20251001"
    ALPHA = 0.5

    tasks = EXAMPLE_TASKS[:4]

    print("Configuration:")
    print(f"  Backend: {BACKEND}")
    print(f"  Model:   {MODEL}")
    print(f"  N_agents={N_AGENTS}, N_rounds={N_ROUNDS}, N_survive={N_SURVIVE}, N_trials={N_TRIALS}")
    print(f"  Tasks: {[t.task_id for t in tasks]}")
    print(f"  Conditions: DDS(alpha={ALPHA}), Debate, Independent")
    print()

    dds_calls = len(tasks) * N_TRIALS * N_AGENTS * N_ROUNDS
    debate_calls = len(tasks) * N_TRIALS * N_AGENTS * N_ROUNDS
    indep_calls = len(tasks) * N_TRIALS * N_AGENTS * 1
    total_calls = dds_calls + debate_calls + indep_calls
    print(f"Estimated API calls: {total_calls:,}")
    print(f"  DDS: {dds_calls:,} | Debate: {debate_calls:,} | Independent: {indep_calls:,}")

    if "--yes" not in sys.argv:
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

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

    all_results = {
        "config": {
            "n_agents": N_AGENTS,
            "n_rounds": N_ROUNDS,
            "n_survive": N_SURVIVE,
            "n_trials": N_TRIALS,
            "backend": BACKEND,
            "model": MODEL,
            "alpha": ALPHA,
            "tasks": [t.task_id for t in tasks],
        },
        "timestamp": datetime.now().isoformat(),
        "conditions": {}
    }

    # ---- DDS ----
    print(f"\n{'='*60}\nCondition: DDS alpha={ALPHA}\n{'='*60}")
    dds_results = []
    for task in tasks:
        task_trials = []
        for trial in range(N_TRIALS):
            print(f"  [{task.task_id}] Trial {trial+1}/{N_TRIALS}...", end="", flush=True)
            try:
                r = run_dds_condition(system, task, ALPHA, N_ROUNDS, N_SURVIVE)
                task_trials.append(r)
                print(f" div={r['final_diversity']:.4f}")
            except Exception as e:
                print(f" ERROR: {e}")
        dds_results.append({"task_id": task.task_id, "trials": task_trials})
    all_results["conditions"][f"dds_alpha_{ALPHA}"] = dds_results

    # ---- Debate ----
    print(f"\n{'='*60}\nCondition: Debate\n{'='*60}")
    debate_results = []
    for task in tasks:
        task_trials = []
        for trial in range(N_TRIALS):
            print(f"  [{task.task_id}] Trial {trial+1}/{N_TRIALS}...", end="", flush=True)
            try:
                r = run_debate_condition(system, task, N_ROUNDS)
                task_trials.append(r)
                print(f" div={r['final_diversity']:.4f}")
            except Exception as e:
                print(f" ERROR: {e}")
        debate_results.append({"task_id": task.task_id, "trials": task_trials})
    all_results["conditions"]["debate"] = debate_results

    # ---- Independent ----
    print(f"\n{'='*60}\nCondition: Independent\n{'='*60}")
    indep_results = []
    for task in tasks:
        task_trials = []
        for trial in range(N_TRIALS):
            print(f"  [{task.task_id}] Trial {trial+1}/{N_TRIALS}...", end="", flush=True)
            try:
                r = run_independent_condition(system, task)
                task_trials.append(r)
                print(f" div={r['final_diversity']:.4f}")
            except Exception as e:
                print(f" ERROR: {e}")
        indep_results.append({"task_id": task.task_id, "trials": task_trials})
    all_results["conditions"]["independent"] = indep_results

    # ---- Save ----
    output_dir = Path("results/robustness_claude")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"robustness_claude_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}\nResults saved to: {output_file}\n{'='*70}")

    # ---- Summary ----
    print("\nSUMMARY:")
    from scipy import stats as scipy_stats
    cond_finals = {}
    for cond_name, cond_data in all_results["conditions"].items():
        all_finals = []
        for task_data in cond_data:
            for trial in task_data["trials"]:
                all_finals.append(trial["final_diversity"])
        cond_finals[cond_name] = all_finals
        n = len(all_finals)
        if n > 0:
            print(f"  {cond_name}: mean={np.mean(all_finals):.4f}, sd={np.std(all_finals, ddof=1):.4f} (n={n})")

    print("\nPaired t-tests (DDS vs others):")
    ref = f"dds_alpha_{ALPHA}"
    if ref in cond_finals:
        for cond, vals in cond_finals.items():
            if cond == ref or len(vals) == 0:
                continue
            n_pair = min(len(cond_finals[ref]), len(vals))
            if n_pair < 3:
                continue
            t, p = scipy_stats.ttest_rel(cond_finals[ref][:n_pair], vals[:n_pair])
            diff = np.array(cond_finals[ref][:n_pair]) - np.array(vals[:n_pair])
            d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {ref} vs {cond}: t={t:.3f}, p={p:.4f}, d={d:.3f} ({sig})")

    print(f"\nCompleted at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Adaptive-Alpha Experiment: Designing Exploration Dynamics

Tests whether a *scheduled* or *feedback-adaptive* selection pressure alpha
produces a better cumulative-exploration trajectory than any fixed alpha.
This is the "design" component of the exploration-dynamics reframing.

Conditions (alpha schedule over the 5 selection steps of a 6-round run):
  1. const      : alpha = 0.5 (best fixed value from the alpha sweep)
  2. increasing : 0.3, 0.5, 0.7, 0.9, 1.1  (spread harder as easy gains exhaust)
  3. decreasing : 1.1, 0.9, 0.7, 0.5, 0.3  (spread early, consolidate late)
  4. adaptive   : start 0.5; if cumulative-diversity gain < eps, alpha += 0.25
                  (cap 1.5) -- closed-loop controller that pushes when exploration stalls

Settings: N=8 agents, 6 rounds, 4 tasks, 5 trials -> 20 obs per condition.
GPT-4o-mini, fixed Q=0.75 to isolate the exploration dynamics.
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
from src.density_selection import (
    DensityDependentSelector, SelectionConfig, AgentResponse
)
from sklearn.metrics.pairwise import cosine_distances

N_AGENTS = 8
N_ROUNDS = 6           # round 0 + 5 selection steps
N_SURVIVE = 5
N_TRIALS = 5
BETA = 2.0
BANDWIDTH = 0.3
ADAPT_EPS = 0.005      # cumulative-diversity gain below this -> increase alpha
ADAPT_STEP = 0.25
ADAPT_CAP = 1.5
ADAPT_START = 0.5

SCHEDULES = {
    "const":      [0.5, 0.5, 0.5, 0.5, 0.5],
    "increasing": [0.3, 0.5, 0.7, 0.9, 1.1],
    "decreasing": [1.1, 0.9, 0.7, 0.5, 0.3],
    # "adaptive" is computed online (see run_condition)
}


def cumulative_diversity(cumulative_pool):
    """Mean pairwise cosine distance over all pooled embeddings so far."""
    arr = np.array(cumulative_pool)
    if len(arr) < 2:
        return 0.0
    d = cosine_distances(arr)
    triu = d[np.triu_indices(len(arr), k=1)]
    return float(np.mean(triu))


def snapshot_diversity(embs):
    arr = np.array(embs)
    if len(arr) < 2:
        return 0.0
    d = cosine_distances(arr)
    triu = d[np.triu_indices(len(arr), k=1)]
    return float(np.mean(triu))


def run_condition(system, task, schedule_name):
    """
    Run a 6-round DDS trajectory under a given alpha schedule.
    Returns per-round cumulative diversity, snapshot diversity, novel fraction,
    the alpha actually used at each selection step, and response texts.
    """
    selector = DensityDependentSelector(SelectionConfig(
        alpha=ADAPT_START, beta=BETA, bandwidth=BANDWIDTH
    ))
    system.reset_all()

    # Round 0: initial generation
    responses = system.generate_responses(task)
    for r in responses:
        r.quality_score = 0.75

    embeddings_by_round = [[r.embedding for r in responses]]
    response_texts = [[{"agent_id": r.agent_id, "text": r.text} for r in responses]]

    # round-0 novelty threshold (median internal distance)
    round0_embs = np.array(embeddings_by_round[0])
    r0d = cosine_distances(round0_embs)
    novelty_threshold = float(np.median(r0d[np.triu_indices(len(round0_embs), k=1)]))

    # online cumulative diversity tracking (for adaptive)
    cumulative_pool = list(embeddings_by_round[0])
    cum_div_prev = cumulative_diversity(cumulative_pool)

    alphas_used = []
    adaptive_alpha = ADAPT_START

    for round_idx in range(1, N_ROUNDS):
        # Determine alpha for this selection step
        if schedule_name == "adaptive":
            alpha = adaptive_alpha
        else:
            alpha = SCHEDULES[schedule_name][round_idx - 1]
        selector.config.alpha = alpha
        alphas_used.append(alpha)

        # Density-dependent selection on current responses
        agent_responses = [
            AgentResponse(
                agent_id=r.agent_id,
                response_text=r.text,
                embedding=r.embedding,
                quality_score=r.quality_score,
                generation=round_idx - 1
            )
            for r in responses
        ]
        selected, _ = selector.select(agent_responses, n_select=N_SURVIVE)

        def context_provider(agent_id, _sel=selected):
            return "\n\n".join(f"Selected response: {s.response_text}" for s in _sel)

        dds_task = Task(
            task_id=f"{task.task_id}_round{round_idx}",
            prompt=f"{task.prompt}\n\nBuild upon or differentiate from the context.",
            category=task.category
        )
        responses = system.generate_responses(dds_task, context_provider)
        for r in responses:
            r.quality_score = 0.75

        embeddings_by_round.append([r.embedding for r in responses])
        response_texts.append([{"agent_id": r.agent_id, "text": r.text} for r in responses])

        # Update cumulative pool + adaptive controller
        cumulative_pool.extend(embeddings_by_round[-1])
        cum_div_now = cumulative_diversity(cumulative_pool)
        gain = cum_div_now - cum_div_prev
        cum_div_prev = cum_div_now

        if schedule_name == "adaptive" and gain < ADAPT_EPS:
            adaptive_alpha = min(ADAPT_CAP, adaptive_alpha + ADAPT_STEP)

    # Compute per-round metrics
    cum_divs, snap_divs, novel_fracs = [], [], []
    pool = []
    for r in range(len(embeddings_by_round)):
        embs = embeddings_by_round[r]
        pool.extend(embs)
        cum_divs.append(cumulative_diversity(pool))
        snap_divs.append(snapshot_diversity(embs))
        if r > 0:
            cross = cosine_distances(np.array(embs), round0_embs)
            min_d = np.min(cross, axis=1)
            novel_fracs.append(float(np.mean(min_d > novelty_threshold)))
        else:
            novel_fracs.append(0.0)

    return {
        "schedule": schedule_name,
        "alphas_used": alphas_used,
        "cumulative_diversity": cum_divs,
        "snapshot_diversity": snap_divs,
        "novel_fraction": novel_fracs,
        "final_cumulative": cum_divs[-1],
        "response_texts": response_texts,
    }


def main():
    print("=" * 70)
    print("ADAPTIVE-ALPHA EXPERIMENT: Designing Exploration Dynamics")
    print("=" * 70)
    print(f"Start: {datetime.now().isoformat()}")

    global N_TRIALS
    if os.environ.get("ADAPTIVE_TRIALS"):
        N_TRIALS = int(os.environ["ADAPTIVE_TRIALS"])
    tasks = EXAMPLE_TASKS[:4]
    if os.environ.get("ADAPTIVE_CONDS"):
        conditions = os.environ["ADAPTIVE_CONDS"].split(",")
    else:
        conditions = ["const", "increasing", "decreasing", "adaptive"]
    total = len(conditions) * len(tasks) * N_TRIALS * N_AGENTS * N_ROUNDS
    print(f"Conditions: {conditions}")
    print(f"N={N_AGENTS}, rounds={N_ROUNDS}, tasks={len(tasks)}, trials={N_TRIALS}")
    print(f"Estimated API calls: {total:,}  (~{total*2/60:.0f} min)")
    if "--yes" not in sys.argv:
        if input("Continue? [y/N]: ").lower() != "y":
            print("Cancelled."); return

    agent_config = AgentConfig(agent_id=0, backend="openai", model="gpt-4o-mini",
                               temperature=0.7, max_tokens=512)
    system = MultiAgentSystem(n_agents=N_AGENTS, agent_config_template=agent_config,
                              embedding_model="all-MiniLM-L6-v2")

    results = {
        "config": {"n_agents": N_AGENTS, "n_rounds": N_ROUNDS, "n_survive": N_SURVIVE,
                   "n_trials": N_TRIALS, "model": "gpt-4o-mini",
                   "schedules": SCHEDULES, "adapt": {"eps": ADAPT_EPS, "step": ADAPT_STEP,
                   "cap": ADAPT_CAP, "start": ADAPT_START},
                   "tasks": [t.task_id for t in tasks]},
        "timestamp": datetime.now().isoformat(),
        "conditions": {},
    }

    for cond in conditions:
        print(f"\n{'='*60}\nCondition: {cond}\n{'='*60}")
        cond_results = []
        for task in tasks:
            trials = []
            for trial in range(N_TRIALS):
                print(f"  [{task.task_id}] trial {trial+1}/{N_TRIALS}...", end="", flush=True)
                try:
                    r = run_condition(system, task, cond)
                    trials.append(r)
                    print(f" cum={r['final_cumulative']:.4f} alphas={r['alphas_used']}")
                except Exception as e:
                    print(f" ERROR: {e}")
            cond_results.append({"task_id": task.task_id, "trials": trials})
        results["conditions"][cond] = cond_results

    out_dir = Path("results/adaptive_alpha")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"adaptive_alpha_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_file}")

    # Summary
    print(f"\n{'='*70}\nSUMMARY: final cumulative diversity by condition\n{'='*70}")
    from scipy import stats as st
    finals = {}
    for cond, data in results["conditions"].items():
        vals = [t["final_cumulative"] for td in data for t in td["trials"]]
        finals[cond] = vals
        if vals:
            print(f"  {cond:12s}: {np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f} (n={len(vals)})")

    print("\nPaired t-tests vs const (alpha=0.5):")
    ref = finals.get("const", [])
    for cond, vals in finals.items():
        if cond == "const" or not vals:
            continue
        n = min(len(ref), len(vals))
        t, p = st.ttest_rel(ref[:n], vals[:n])
        diff = np.array(vals[:n]) - np.array(ref[:n])
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        arrow = "higher" if np.mean(vals[:n]) > np.mean(ref[:n]) else "lower"
        print(f"  {cond} vs const: {arrow}, t={t:.3f}, p={p:.4f}, d={d:.3f} ({sig})")

    print(f"\nDone: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

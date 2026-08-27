#!/usr/bin/env python3
"""
M4: the missing factorial cell -- DDS selection at a HIGH decoding temperature.

Selection was tested only at T=0.7 and temperature only with independent sampling,
so we never saw how selection behaves on the wider, higher-dimensional manifold
that a high temperature opens. This runs the SAME DDS dynamics as
run_dynamics_with_mapelites.py (alpha=0.5, N=8, 3 rounds, 5 trials, 4 tasks,
gpt-4o-mini) but with decoding temperature 1.2 instead of 0.7. The output format
matches the dynamics file so the existing d_eff / subspace analyses load it
directly. The question: does selection keep the temperature-1.2 effective dimension
(harvest tails within the wider manifold) or pull it back toward the T=0.7 level
(re-confine)?
"""

import sys, os, json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_agent import MultiAgentSystem, AgentConfig, EXAMPLE_TASKS
from src.protocols import DDSProtocol, DDSConfig


def compute_diversity(responses):
    import numpy as np
    embs = [r.embedding for r in responses if getattr(r, "embedding", None) is not None]
    if len(embs) < 2:
        return 0.0
    E = np.asarray(embs, dtype=float)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    n = len(E); s = 0.0; c = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += 1.0 - float(E[i] @ E[j]); c += 1
    return s / max(c, 1)


def extract_texts(responses):
    return [{"agent_id": getattr(r, "agent_id", i), "text": r.text}
            for i, r in enumerate(responses)]


def run_dds_condition(system, task, alpha, n_rounds, n_survive):
    config = DDSConfig(n_rounds=n_rounds, n_agents=system.n_agents, alpha=alpha,
                       beta=2.0, bandwidth=0.3, n_survive=n_survive)
    result = DDSProtocol(config).run(system, task, quality_evaluator=None)
    diversities, texts = [], []
    for round_responses in result["round_history"]:
        diversities.append(compute_diversity(round_responses))
        texts.append(extract_texts(round_responses))
    return {"round_diversities": diversities, "final_diversity": diversities[-1],
            "n_rounds": len(diversities), "response_texts": texts}


def main():
    N_AGENTS, N_ROUNDS, N_SURVIVE, N_TRIALS = 8, 3, 5, 5
    ALPHA, TEMP = 0.5, 1.2
    tasks = EXAMPLE_TASKS[:4]
    calls = len(tasks) * N_TRIALS * N_AGENTS * N_ROUNDS
    print(f"DDS alpha={ALPHA} at decoding temperature {TEMP}")
    print(f"N={N_AGENTS}, rounds={N_ROUNDS}, trials={N_TRIALS}, tasks={[t.task_id for t in tasks]}")
    print(f"~{calls} calls, ~${calls*0.0002:.2f}, ~{calls*2/60:.0f} min\n")

    agent_config = AgentConfig(agent_id=0, backend="openai", model="gpt-4o-mini",
                               temperature=TEMP, max_tokens=512)
    system = MultiAgentSystem(n_agents=N_AGENTS, agent_config_template=agent_config,
                              embedding_model="all-MiniLM-L6-v2")

    out = {"config": {"n_agents": N_AGENTS, "n_rounds": N_ROUNDS, "n_survive": N_SURVIVE,
                      "n_trials": N_TRIALS, "backend": "openai", "model": "gpt-4o-mini",
                      "alpha": ALPHA, "temperature": TEMP,
                      "tasks": [t.task_id for t in tasks]},
           "timestamp": datetime.now().isoformat(), "conditions": {}}

    cond = []
    for task in tasks:
        trials = []
        for trial in range(N_TRIALS):
            print(f"  [{task.task_id}] trial {trial+1}/{N_TRIALS}...", end="", flush=True)
            try:
                r = run_dds_condition(system, task, ALPHA, N_ROUNDS, N_SURVIVE)
                trials.append(r); print(f" div={r['final_diversity']:.4f}")
            except Exception as e:
                print(f" ERROR: {e}")
        cond.append({"task_id": task.task_id, "trials": trials})
    out["conditions"][f"dds_alpha_{ALPHA}"] = cond

    Path("results/temperature_expansion").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"results/temperature_expansion/dds_temp12_{stamp}.json"
    Path(fn).write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {fn}")


if __name__ == "__main__":
    main()

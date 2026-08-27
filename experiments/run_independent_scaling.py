#!/usr/bin/env python3
"""
Independent generation at large N (single round, no interaction).

This samples the LLM's raw output support for each task, independent of any
multi-agent regeneration loop. Comparing this support to what the DDS loop
reaches (its plateau) separates two hypotheses for the exploration ceiling:
  Pattern A: DDS reaches the full support -> plateau = generator/manifold ceiling.
  Pattern B: independent support >> DDS reach -> the regeneration (ICL) loop
             confines exploration to a sub-region ("ICL dynamic trap").

We generate N=128 independent responses per task and save the texts. The
coverage/effective-dimension analysis is in analyze_support_vs_loop.py.
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_agent import MultiAgentSystem, AgentConfig, EXAMPLE_TASKS

N_INDEP = 128
MODEL = "gpt-4o-mini"


def main():
    print("=" * 70)
    print("INDEPENDENT GENERATION AT LARGE N (raw output support)")
    print("=" * 70)
    print(f"Start: {datetime.now().isoformat()}")
    tasks = EXAMPLE_TASKS[:4]
    total = N_INDEP * len(tasks)
    print(f"N_indep={N_INDEP} per task, tasks={[t.task_id for t in tasks]}")
    print(f"Estimated API calls: {total:,} (~{total*3/60:.0f} min)")
    if "--yes" not in sys.argv:
        if input("Continue? [y/N]: ").lower() != "y":
            print("Cancelled."); return

    cfg = AgentConfig(agent_id=0, backend="openai", model=MODEL,
                      temperature=0.7, max_tokens=512)
    # One system with N_INDEP agents; a single generate_responses call = N_INDEP
    # independent responses (no context provider -> no interaction).
    system = MultiAgentSystem(n_agents=N_INDEP, agent_config_template=cfg,
                              embedding_model="all-MiniLM-L6-v2")

    results = {
        "config": {"n_indep": N_INDEP, "model": MODEL,
                   "tasks": [t.task_id for t in tasks]},
        "timestamp": datetime.now().isoformat(),
        "responses_by_task": {},
    }

    out_dir = Path("results/independent_scaling")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"independent_scaling_{ts}.json"

    for task in tasks:
        print(f"\n[{task.task_id}] generating {N_INDEP} independent responses...")
        responses = system.generate_responses(task)
        texts = [{"agent_id": r.agent_id, "text": r.text} for r in responses]
        results["responses_by_task"][task.task_id] = texts
        print(f"  got {len(texts)} responses")
        # Incremental save (resilient to interruption)
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\nSaved: {out_file}")
    print(f"Done: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

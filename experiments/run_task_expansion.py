#!/usr/bin/env python3
"""Task-expansion runner: one task through conditions A-F of
paper/task_expansion_plan.md (pilot and full run share this script).

Usage: python3 run_task_expansion.py <task_id>

Conditions per task:
  A/B  indep_t07, indep_t10, indep_t12   (N=128 independent, single round)
  F    prompt_v1, prompt_v2              (N=128 independent at T=0.7 + suffix)
  C/D  dds_a05_t07, dds_a05_t12          (DDS alpha=0.5, 5 trials, 8x3 rounds)
  E    debate_t07, map_elites_t07        (5 trials each, 8x3 rounds)

Resumable: JSON is dumped after every trial / every 32 independent responses;
rerunning skips completed units. Output: results/task_expansion/pilot_<task>_*.json
"""
import sys
import os
import json
import glob
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_agent import MultiAgentSystem, AgentConfig, EXAMPLE_TASKS
from run_dynamics_with_mapelites import (
    run_dds_condition, run_debate_condition, run_mapelites_condition
)

MODEL = "gpt-4o-mini"
MAX_TOKENS = 512
N_INDEP = 128
N_AGENTS = 8
N_ROUNDS = 3
N_SURVIVE = 5
N_TRIALS = 5
RETRIES = 6

# Same wordings as run_prompt_expansion.py / run_prompt_variant.py.
SUFFIX_V1 = ("\n\nImportant: give a response that takes a distinctive, non-obvious "
             "angle, deliberately different from the most common or expected answer.")
SUFFIX_V2 = ("\n\nApproach this from an unconventional angle: give an answer that most "
             "people would not think of, one that stands apart from the typical or "
             "default response.")


def gen_retry(agent, prompt):
    delay = 2.0
    for a in range(RETRIES):
        try:
            return agent.generate(prompt)
        except Exception as e:
            if a == RETRIES - 1:
                raise
            print(f"    retry {a+1}: {type(e).__name__}; wait {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30)


def main():
    task_id = sys.argv[1]
    task = next(t for t in EXAMPLE_TASKS if t.task_id == task_id)
    out_dir = Path("results/task_expansion")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume from the latest file for this task, if any.
    old = sorted(glob.glob(str(out_dir / f"pilot_{task_id}_2*.json")))
    if old:
        res = json.load(open(old[-1]))
        print(f"Resuming from {old[-1]}", flush=True)
    else:
        res = {
            "config": {"model": MODEL, "max_tokens": MAX_TOKENS, "n_indep": N_INDEP,
                       "n_agents": N_AGENTS, "n_rounds": N_ROUNDS,
                       "n_survive": N_SURVIVE, "n_trials": N_TRIALS,
                       "task_id": task_id, "task_prompt": task.prompt,
                       "category": task.category,
                       "suffix_v1": SUFFIX_V1, "suffix_v2": SUFFIX_V2},
            "timestamp": datetime.now().isoformat(),
            "independent": {},
            "loops": {},
        }
    of = out_dir / f"pilot_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def save():
        json.dump(res, open(of, "w"), indent=1)

    save()
    print(f"TASK EXPANSION [{task_id}] ({task.category})", flush=True)
    print(f"Start: {datetime.now().isoformat()}", flush=True)

    # ---------------- A/B/F: independent batches ----------------
    batch_specs = [
        ("indep_t07", 0.7, ""),
        ("indep_t10", 1.0, ""),
        ("indep_t12", 1.2, ""),
        ("prompt_v1", 0.7, SUFFIX_V1),
        ("prompt_v2", 0.7, SUFFIX_V2),
    ]
    for name, temp, suffix in batch_specs:
        got = res["independent"].get(name, [])
        if len(got) >= N_INDEP:
            print(f"[{name}] complete, skip", flush=True)
            continue
        print(f"[{name}] generating {N_INDEP - len(got)} (T={temp})...", flush=True)
        cfg = AgentConfig(agent_id=0, backend="openai", model=MODEL,
                          temperature=temp, max_tokens=MAX_TOKENS)
        system = MultiAgentSystem(n_agents=N_INDEP, agent_config_template=cfg,
                                  embedding_model="all-MiniLM-L6-v2")
        prompt = task.prompt + suffix
        texts = list(got)
        for i in range(len(got), N_INDEP):
            texts.append({"agent_id": i, "text": gen_retry(system.agents[i], prompt)})
            if (i + 1) % 32 == 0:
                print(f"    {i+1}/{N_INDEP}", flush=True)
                res["independent"][name] = texts
                save()
        res["independent"][name] = texts
        save()

    # ---------------- C/D/E: loop conditions ----------------
    def loop_system(temp):
        cfg = AgentConfig(agent_id=0, backend="openai", model=MODEL,
                          temperature=temp, max_tokens=MAX_TOKENS)
        return MultiAgentSystem(n_agents=N_AGENTS, agent_config_template=cfg,
                                embedding_model="all-MiniLM-L6-v2")

    loop_specs = [
        ("dds_a05_t07", 0.7, "dds"),
        ("dds_a05_t12", 1.2, "dds"),
        ("debate_t07", 0.7, "debate"),
        ("map_elites_t07", 0.7, "map"),
    ]
    for name, temp, kind in loop_specs:
        trials = res["loops"].get(name, [])
        if len(trials) >= N_TRIALS:
            print(f"[{name}] complete, skip", flush=True)
            continue
        system = loop_system(temp)
        for k in range(len(trials), N_TRIALS):
            print(f"[{name}] trial {k+1}/{N_TRIALS}...", flush=True)
            system.reset_all()
            if kind == "dds":
                tr = run_dds_condition(system, task, 0.5, N_ROUNDS, N_SURVIVE)
            elif kind == "debate":
                tr = run_debate_condition(system, task, N_ROUNDS)
            else:
                tr = run_mapelites_condition(system, task, N_ROUNDS)
            trials.append(tr)
            res["loops"][name] = trials
            save()

    save()
    print(f"\nSaved: {of}\nDone: {datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()

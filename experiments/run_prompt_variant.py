#!/usr/bin/env python3
"""
Robustness of the novelty axis to the exact prompt wording.

The finding that a distinctiveness prompt keeps the effective dimension flat but
reaches the MOST new directions (subspace leakage) rests, in the main experiment,
on one instruction. Here we regenerate the prompt condition with a DIFFERENTLY
WORDED distinctiveness instruction (same intent, different phrasing, same fixed
decoding T=0.7, N=128 per task) so the leakage can be recomputed and the ordering
(prompt > temperature > selection) re-checked. If the new wording leaks like the
original, the novelty finding is not specific to one phrasing.
"""

import sys, os, json, glob, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.llm_agent import MultiAgentSystem, AgentConfig, EXAMPLE_TASKS

N_INDEP, MODEL, MAX_TOKENS, TEMP, RETRIES = 128, "gpt-4o-mini", 512, 0.7, 6
# Differently worded from the original ("distinctive, non-obvious angle,
# deliberately different from the most common or expected answer").
SUFFIX = ("\n\nApproach this from an unconventional angle: give an answer that most "
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
            time.sleep(delay); delay = min(delay * 2, 30)


def main():
    print("PROMPT-VARIANT (differently worded distinctiveness instruction, T=0.7)")
    tasks = EXAMPLE_TASKS[:4]
    out_dir = Path("results/prompt_variant"); out_dir.mkdir(parents=True, exist_ok=True)
    ex = sorted(glob.glob("results/prompt_variant/prompt_variant_2*.json"))
    existing = json.load(open(ex[-1]))["responses_by_task"] if ex else {}
    of = out_dir / f"prompt_variant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results = {"config": {"n_indep": N_INDEP, "model": MODEL, "temperature": TEMP,
                          "prompt_suffix": SUFFIX},
               "timestamp": datetime.now().isoformat(),
               "responses_by_task": dict(existing)}
    cfg = AgentConfig(agent_id=0, backend="openai", model=MODEL,
                      temperature=TEMP, max_tokens=MAX_TOKENS)
    system = None
    for task in tasks:
        if len(results["responses_by_task"].get(task.task_id, [])) >= N_INDEP:
            print(f"[{task.task_id}] complete, skip", flush=True); continue
        if system is None:
            system = MultiAgentSystem(n_agents=N_INDEP, agent_config_template=cfg,
                                      embedding_model="all-MiniLM-L6-v2")
        prompt = task.prompt + SUFFIX
        print(f"[{task.task_id}] generating {N_INDEP}...", flush=True)
        texts = []
        for i, agent in enumerate(system.agents):
            texts.append({"agent_id": agent.config.agent_id,
                          "text": gen_retry(agent, prompt)})
            if (i + 1) % 32 == 0:
                print(f"    {i+1}/{N_INDEP}", flush=True)
        results["responses_by_task"][task.task_id] = texts
        json.dump(results, open(of, "w"), indent=2)
    json.dump(results, open(of, "w"), indent=2)
    print(f"\nSaved: {of}", flush=True)


if __name__ == "__main__":
    main()

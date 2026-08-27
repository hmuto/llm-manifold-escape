#!/usr/bin/env python3
"""
Prompt-lever expansion: does a prompt-level intervention (input from outside the
closed loop, at the SAME fixed decoding) add dimensions the way raising the
temperature does?

We generate N=128 independent responses per task at temperature 0.7 (the same
decoding as the reference) but with a distinctiveness instruction appended to the
prompt (a verbalized-sampling-style, external prompt change). The effective
dimension / escape / quality analysis mirrors the temperature experiment, giving
a second, independent test of "external input adds dimensions." Retry+resume.
"""

import sys, os, json, glob, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.llm_agent import MultiAgentSystem, AgentConfig, Task, EXAMPLE_TASKS

N_INDEP = 128
MODEL = "gpt-4o-mini"
MAX_TOKENS = 512
TEMP = 0.7
RETRIES = 6
SUFFIX = ("\n\nImportant: give a response that takes a distinctive, non-obvious "
          "angle, deliberately different from the most common or expected answer.")


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


def load_existing():
    fs = sorted(glob.glob("results/prompt_expansion/prompt_expansion_2*.json"))
    return json.load(open(fs[-1]))["responses_by_task"] if fs else {}


def main():
    print("PROMPT-LEVER EXPANSION (external prompt input at fixed decoding T=0.7)")
    print(f"Start: {datetime.now().isoformat()}", flush=True)
    tasks = EXAMPLE_TASKS[:4]
    existing = load_existing()
    print(f"Resuming: {len(existing)} tasks already complete", flush=True)

    out_dir = Path("results/prompt_expansion"); out_dir.mkdir(parents=True, exist_ok=True)
    of = out_dir / f"prompt_expansion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results = {"config": {"n_indep": N_INDEP, "model": MODEL, "temperature": TEMP,
                          "max_tokens": MAX_TOKENS, "prompt_suffix": SUFFIX},
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
        print(f"[{task.task_id}] generating {N_INDEP} (prompt+suffix)...", flush=True)
        texts = []
        for i, agent in enumerate(system.agents):
            texts.append({"agent_id": agent.config.agent_id,
                          "text": gen_retry(agent, prompt)})
            if (i + 1) % 32 == 0:
                print(f"    {i+1}/{N_INDEP}", flush=True)
        results["responses_by_task"][task.task_id] = texts
        json.dump(results, open(of, "w"), indent=2)
        print(f"  got {len(texts)}", flush=True)

    json.dump(results, open(of, "w"), indent=2)
    print(f"\nSaved: {of}\nDone: {datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()

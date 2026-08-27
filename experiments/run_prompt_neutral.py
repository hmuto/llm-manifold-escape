#!/usr/bin/env python3
"""Neutral-prompt control for directional novelty.

The distinctiveness prompt asks for semantically different content. A natural
objection is that ANY added instruction rotates the output distribution. This
control adds an instruction that is semantically neutral (politeness and
format only) and measures the same three geometric quantities. If its leakage
stays at the control level while the distinctiveness prompt's does not, the
directional effect is attributable to the instruction's semantic content.

Twelve tasks, N=128 independent responses at T=0.7, gpt-4o-mini (matching the
prompt_v1 protocol; only the suffix differs). Resume-safe.
"""
import sys, os, json, glob, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.llm_agent import MultiAgentSystem, AgentConfig, EXAMPLE_TASKS

N_INDEP, MODEL, MAX_TOKENS, RETRIES = 128, "gpt-4o-mini", 512, 6
SUFFIX = ("\n\nImportant: answer politely, and structure your response in "
          "exactly two paragraphs.")
OUT_DIR = Path("results/prompt_neutral")


def gen_retry(agent, prompt):
    delay = 2.0
    for a in range(RETRIES):
        try:
            return agent.generate(prompt)
        except Exception as e:
            if a == RETRIES - 1:
                raise
            print(f"    retry {a+1} after {type(e).__name__}", flush=True)
            time.sleep(delay); delay = min(delay * 2, 30.0)


def load_existing():
    fs = sorted(glob.glob(str(OUT_DIR / "prompt_neutral_2*.json")))
    return json.load(open(fs[-1]))["responses_by_task"] if fs else {}


def main():
    # EXAMPLE_TASKS contains 16 entries; the four code_* tasks are generated
    # here for completeness but are NOT part of the paper's twelve canonical
    # tasks. analyze_prompt_neutral.py filters to the canonical twelve.
    tasks = EXAMPLE_TASKS
    done = load_existing()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    of = OUT_DIR / f"prompt_neutral_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    res = {"config": {"n_indep": N_INDEP, "model": MODEL, "temperature": 0.7,
                      "max_tokens": MAX_TOKENS, "prompt_suffix": SUFFIX},
           "timestamp": datetime.now().isoformat(),
           "responses_by_task": dict(done)}

    def save():
        json.dump(res, open(of, "w"), indent=1, default=str)

    cfg = AgentConfig(agent_id=0, backend="openai", model=MODEL,
                      temperature=0.7, max_tokens=MAX_TOKENS)
    system = MultiAgentSystem(n_agents=N_INDEP, agent_config_template=cfg,
                              embedding_model="all-MiniLM-L6-v2")
    for task in tasks:
        texts = res["responses_by_task"].get(task.task_id, [])
        if len(texts) >= N_INDEP:
            print(f"[{task.task_id}] complete, skip", flush=True)
            continue
        print(f"[{task.task_id}] generating {N_INDEP}...", flush=True)
        for i, agent in enumerate(system.agents[len(texts):]):
            texts.append({"agent_id": agent.config.agent_id,
                          "text": gen_retry(agent, task.prompt + SUFFIX)})
            if len(texts) % 32 == 0:
                res["responses_by_task"][task.task_id] = texts; save()
                print(f"    {len(texts)}/{N_INDEP}", flush=True)
        res["responses_by_task"][task.task_id] = texts; save()
    save()
    print(f"Saved: {of}\nDone: {datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()

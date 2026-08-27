#!/usr/bin/env python3
"""
Temperature expansion: can decoding lift the ceiling that selection cannot?

The support analysis (analyze_support_vs_loop.py) shows that DDS selection at
temperature 0.7 does not exceed the accessible support estimated by independent
N=128 sampling at temperature 0.7. That support, however, is the support *at a
fixed decoding*. Here we test whether simply raising the decoding temperature
lets independent generation reach regions outside the temperature-0.7 support.

We generate N=128 independent responses per task at temperature 1.0 and 1.2
(same model, max_tokens, and embedding as the temperature-0.7 reference; only the
temperature changes) and save the texts. The escape/radius analysis is in
analyze_temperature_expansion.py.

Robustness: each response is generated with retry+backoff, and the run resumes
from the latest saved file (already-completed temp/task combos are reused), so a
transient API connection error does not lose completed work.
"""

import sys, os, json, glob, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_agent import MultiAgentSystem, AgentConfig, EXAMPLE_TASKS

N_INDEP = 128
MODEL = "gpt-4o-mini"
MAX_TOKENS = 512                 # match the temperature-0.7 reference
TEMPERATURES = [1.0, 1.2]
RETRIES = 6                      # per-response retries on transient errors


def generate_with_retry(agent, prompt):
    delay = 2.0
    for attempt in range(RETRIES):
        try:
            return agent.generate(prompt)
        except Exception as e:
            if attempt == RETRIES - 1:
                raise
            print(f"    retry {attempt+1}/{RETRIES-1} after error: "
                  f"{type(e).__name__}; waiting {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def load_existing():
    """Resume: reuse completed temp/task combos from the latest saved file."""
    files = sorted(glob.glob("results/temperature_expansion/temperature_expansion_2*.json"))
    if not files:
        return {}
    d = json.load(open(files[-1]))
    return d.get("responses_by_temp_task", {})


def main():
    print("=" * 70)
    print("TEMPERATURE EXPANSION (does decoding exceed the fixed-decoding support?)")
    print("=" * 70)
    print(f"Start: {datetime.now().isoformat()}", flush=True)
    tasks = EXAMPLE_TASKS[:4]
    print(f"N_indep={N_INDEP}, temps={TEMPERATURES}, "
          f"tasks={[t.task_id for t in tasks]}", flush=True)

    existing = load_existing()
    n_done = sum(len(v) for v in existing.values())
    print(f"Resuming: {n_done} temp/task combos already complete", flush=True)

    out_dir = Path("results/temperature_expansion")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"temperature_expansion_{ts}.json"

    results = {
        "config": {"n_indep": N_INDEP, "model": MODEL, "max_tokens": MAX_TOKENS,
                   "temperatures": TEMPERATURES, "tasks": [t.task_id for t in tasks]},
        "timestamp": datetime.now().isoformat(),
        "responses_by_temp_task": {f"temp_{t}": dict(existing.get(f"temp_{t}", {}))
                                   for t in TEMPERATURES},
    }

    def save():
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

    for temp in TEMPERATURES:
        key = f"temp_{temp}"
        cfg = AgentConfig(agent_id=0, backend="openai", model=MODEL,
                          temperature=temp, max_tokens=MAX_TOKENS)
        system = None
        for task in tasks:
            done = results["responses_by_temp_task"][key].get(task.task_id, [])
            if len(done) >= N_INDEP:
                print(f"[temp={temp}] [{task.task_id}] already complete ({len(done)}), skip",
                      flush=True)
                continue
            if system is None:
                system = MultiAgentSystem(n_agents=N_INDEP, agent_config_template=cfg,
                                          embedding_model="all-MiniLM-L6-v2")
            print(f"[temp={temp}] [{task.task_id}] generating {N_INDEP} responses...",
                  flush=True)
            texts = []
            for i, agent in enumerate(system.agents):
                text = generate_with_retry(agent, task.prompt)
                texts.append({"agent_id": agent.config.agent_id, "text": text})
                if (i + 1) % 32 == 0:
                    print(f"    {i+1}/{N_INDEP}", flush=True)
            results["responses_by_temp_task"][key][task.task_id] = texts
            print(f"  got {len(texts)} responses", flush=True)
            save()

    save()
    print(f"\nSaved: {out_file}")
    print(f"Done: {datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()

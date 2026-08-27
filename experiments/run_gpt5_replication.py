#!/usr/bin/env python3
"""
GPT-5 replication: do the three geometric verdicts hold on a frontier-scale
generator?

Runs the four key conditions on the original four tasks with
gpt-5-chat-latest (the non-reasoning GPT-5 chat variant; it accepts the
temperature parameter, unlike the reasoning gpt-5 models):

  ref07     independent N=128 at T=0.7   (defines the GPT-5 reference region)
  temp12    independent N=128 at T=1.2   (dimensional-expansion lever)
  prompt_v1 independent N=128 at T=0.7 + distinctiveness suffix (directional lever)
  dds07     DDS closed loop, N=8, 3 rounds, 5 trials, alpha=0.5, constant Q

All comparisons are within-model (GPT-5 conditions vs the GPT-5 ref07), matching
the paper's design. Analysis (d_eff / out-of-reference rate / leakage) is in
analyze_gpt5_replication.py.

Resume-safe: progress is saved after every condition/task and every DDS trial;
rerunning reuses the latest saved file.
"""

import sys, os, json, glob, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_agent import MultiAgentSystem, AgentConfig, EXAMPLE_TASKS
from src.protocols import DDSProtocol, DDSConfig

MODEL = "gpt-5-chat-latest"
N_INDEP = 128
MAX_TOKENS = 512
N_AGENTS, N_ROUNDS, N_SURVIVE, N_TRIALS, ALPHA = 8, 3, 5, 5, 0.5
RETRIES = 6
SUFFIX = ("\n\nImportant: give a response that takes a distinctive, non-obvious "
          "angle, deliberately different from the most common or expected answer.")

OUT_DIR = Path("results/gpt5_replication")


def served_model():
    """Record which snapshot the -latest alias resolves to at run time."""
    from openai import OpenAI
    r = OpenAI().chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "OK"}], max_tokens=4)
    return r.model


def gen_retry(agent, prompt):
    delay = 2.0
    for attempt in range(RETRIES):
        try:
            return agent.generate(prompt)
        except Exception as e:
            if attempt == RETRIES - 1:
                raise
            print(f"    retry {attempt+1}/{RETRIES-1} after {type(e).__name__}; "
                  f"waiting {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def load_existing():
    files = sorted(glob.glob(str(OUT_DIR / "gpt5_replication_2*.json")))
    if not files:
        return {}, {}
    d = json.load(open(files[-1]))
    return d.get("responses", {}), d.get("dds", {})


def main():
    print("=" * 70)
    print("GPT-5 REPLICATION (frontier-scale check of the three verdicts)")
    print("=" * 70)
    print(f"Start: {datetime.now().isoformat()}", flush=True)
    tasks = EXAMPLE_TASKS[:4]
    sm = served_model()
    print(f"model={MODEL} (served: {sm})", flush=True)
    print(f"tasks={[t.task_id for t in tasks]}", flush=True)

    indep_done, dds_done = load_existing()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"gpt5_replication_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    results = {
        "config": {"model": MODEL, "served_model": sm, "max_tokens": MAX_TOKENS,
                   "n_indep": N_INDEP, "prompt_suffix": SUFFIX,
                   "dds": {"n_agents": N_AGENTS, "n_rounds": N_ROUNDS,
                           "n_survive": N_SURVIVE, "n_trials": N_TRIALS,
                           "alpha": ALPHA, "beta": 2.0, "bandwidth": 0.3},
                   "tasks": [t.task_id for t in tasks]},
        "timestamp": datetime.now().isoformat(),
        "responses": {c: dict(indep_done.get(c, {}))
                      for c in ("ref07", "temp12", "prompt_v1")},
        "dds": {t: list(dds_done.get(t, [])) for t in [x.task_id for x in tasks]},
    }

    def save():
        with open(out_file, "w") as f:
            json.dump(results, f, indent=1, default=str)

    # ---------- independent conditions ----------
    INDEP = [("ref07", 0.7, ""), ("temp12", 1.2, ""), ("prompt_v1", 0.7, SUFFIX)]
    for cond, temp, suffix in INDEP:
        cfg = AgentConfig(agent_id=0, backend="openai", model=MODEL,
                          temperature=temp, max_tokens=MAX_TOKENS)
        system = None
        for task in tasks:
            done = results["responses"][cond].get(task.task_id, [])
            if len(done) >= N_INDEP:
                print(f"[{cond}] [{task.task_id}] complete ({len(done)}), skip", flush=True)
                continue
            if system is None:
                system = MultiAgentSystem(n_agents=N_INDEP, agent_config_template=cfg,
                                          embedding_model="all-MiniLM-L6-v2")
            print(f"[{cond}] [{task.task_id}] generating {N_INDEP}...", flush=True)
            texts = list(done)
            for i, agent in enumerate(system.agents[len(texts):]):
                texts.append({"agent_id": agent.config.agent_id,
                              "text": gen_retry(agent, task.prompt + suffix)})
                if len(texts) % 32 == 0:
                    print(f"    {len(texts)}/{N_INDEP}", flush=True)
                    results["responses"][cond][task.task_id] = texts
                    save()
            results["responses"][cond][task.task_id] = texts
            save()
            print(f"  got {len(texts)}", flush=True)

    # ---------- DDS closed loop ----------
    cfg = AgentConfig(agent_id=0, backend="openai", model=MODEL,
                      temperature=0.7, max_tokens=MAX_TOKENS)
    system = MultiAgentSystem(n_agents=N_AGENTS, agent_config_template=cfg,
                              embedding_model="all-MiniLM-L6-v2")
    dcfg = DDSConfig(n_rounds=N_ROUNDS, n_agents=N_AGENTS, alpha=ALPHA,
                     beta=2.0, bandwidth=0.3, n_survive=N_SURVIVE)
    for task in tasks:
        trials = results["dds"][task.task_id]
        while len(trials) < N_TRIALS:
            t0 = time.time()
            print(f"[dds07] [{task.task_id}] trial {len(trials)+1}/{N_TRIALS}...", flush=True)
            protocol = DDSProtocol(dcfg)
            res = protocol.run(system, task, quality_evaluator=None)
            rounds = [[{"agent_id": r.agent_id, "text": r.text} for r in rr]
                      for rr in res["round_history"]]
            trials.append(rounds)
            save()
            print(f"    done in {time.time()-t0:.0f}s "
                  f"({sum(len(r) for r in rounds)} responses)", flush=True)

    save()
    print(f"\nSaved: {out_file}")
    print(f"Done: {datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()

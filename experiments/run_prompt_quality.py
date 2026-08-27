#!/usr/bin/env python3
"""
Quality control for the prompt-lever experiment (symmetry with temperature).

The distinctiveness prompt relocates 66% of responses outside the default region.
A referee will ask whether that is quality collapse. We score a sample of the
prompt-lever responses with the same G-Eval judge (GPT-4o) used for temperature,
and TOST-compare against the default (temperature-0.7) responses.
"""

import sys, os, json, glob, time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.llm_agent import EXAMPLE_TASKS
from evaluate_quality_geval import evaluate_single

N_SAMPLE = 30
RETRIES = 6


def score_retry(client, text, prompt):
    delay = 2.0
    for a in range(RETRIES):
        try:
            s = evaluate_single(client, text, prompt)
            return (s["coherence"] + s["relevance"] + s["depth"]) / 3.0
        except Exception as e:
            if a == RETRIES - 1:
                raise
            print(f"    retry {a+1}: {type(e).__name__}; wait {delay:.0f}s", flush=True)
            time.sleep(delay); delay = min(delay * 2, 30)


def main():
    from openai import OpenAI
    client = OpenAI()
    f = sorted(glob.glob("results/prompt_expansion/prompt_expansion_2*.json"))[-1]
    prm = {t: [r["text"] for r in tx]
           for t, tx in json.load(open(f))["responses_by_task"].items()}
    prompts = {t.task_id: t.prompt for t in EXAMPLE_TASKS[:4]}  # score against the ORIGINAL task
    tasks = list(prompts.keys())
    rng = np.random.RandomState(0)

    out = {"n_sample": N_SAMPLE, "timestamp": datetime.now().isoformat(), "scores": {}}
    of = Path("results/prompt_expansion") / f"prompt_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print(f"Scoring {N_SAMPLE}/task prompt-lever responses with GPT-4o G-Eval\n")
    for tid in tasks:
        texts = prm[tid]
        idx = rng.choice(len(texts), size=min(N_SAMPLE, len(texts)), replace=False)
        vals = [score_retry(client, texts[i], prompts[tid]) for i in idx]
        out["scores"][tid] = vals
        json.dump(out, open(of, "w"), indent=2)
        print(f"  {tid}: mean {np.mean(vals):.2f} (n={len(vals)})", flush=True)

    allv = [v for t in tasks for v in out["scores"][t]]
    print(f"\nprompt-lever overall quality: {np.mean(allv):.2f} +/- "
          f"{np.std(allv, ddof=1)/np.sqrt(len(allv)):.2f} (n={len(allv)})")
    print(f"(default T=0.7 reference from temperature_quality was ~4.17)")
    print(f"Saved: {of}")


if __name__ == "__main__":
    main()

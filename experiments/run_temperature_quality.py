#!/usr/bin/env python3
"""
Quality control for the temperature-expansion experiment.

The temperature experiment shows that higher decoding temperature enlarges the
accessible region. A referee concern is that the enlargement could be incoherent
output scattering rather than genuine diversity. We therefore score a sample of
the temperature-0.7, 1.0, and 1.2 independent responses with the same G-Eval
judge (GPT-4o, coherence/relevance/depth on 1-5) used elsewhere, and compare mean
quality across temperatures.
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


def load_sets():
    ref_f = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))[-1]
    ref = {tid: [r["text"] for r in rs]
           for tid, rs in json.load(open(ref_f))["responses_by_task"].items()}
    tf = sorted(glob.glob("results/temperature_expansion/temperature_expansion_2*.json"))[-1]
    tmp = json.load(open(tf))["responses_by_temp_task"]
    return {"temp_0.7": ref,
            "temp_1.0": {t: [r["text"] for r in v] for t, v in tmp["temp_1.0"].items()},
            "temp_1.2": {t: [r["text"] for r in v] for t, v in tmp["temp_1.2"].items()}}


def main():
    from openai import OpenAI
    client = OpenAI()
    sets = load_sets()
    prompts = {t.task_id: t.prompt for t in EXAMPLE_TASKS[:4]}
    tasks = list(prompts.keys())
    rng = np.random.RandomState(0)

    out = {"n_sample": N_SAMPLE, "timestamp": datetime.now().isoformat(), "scores": {}}
    out_dir = Path("results/temperature_expansion"); out_dir.mkdir(parents=True, exist_ok=True)
    of = out_dir / f"temperature_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print(f"Scoring {N_SAMPLE}/task/temp with GPT-4o G-Eval\n")
    for temp, per_task in sets.items():
        out["scores"][temp] = {}
        for tid in tasks:
            texts = per_task[tid]
            idx = rng.choice(len(texts), size=min(N_SAMPLE, len(texts)), replace=False)
            vals = []
            for i in idx:
                vals.append(score_retry(client, texts[i], prompts[tid]))
            out["scores"][temp][tid] = vals
            json.dump(out, open(of, "w"), indent=2)
            print(f"  {temp} {tid}: mean {np.mean(vals):.2f} (n={len(vals)})", flush=True)

    print("\n=== MEAN QUALITY (1-5) by temperature ===")
    for temp in sets:
        allv = [v for tid in tasks for v in out["scores"][temp][tid]]
        print(f"  {temp}: {np.mean(allv):.2f} +/- {np.std(allv, ddof=1)/np.sqrt(len(allv)):.2f} (SEM)")
    print(f"\nSaved: {of}")


if __name__ == "__main__":
    main()

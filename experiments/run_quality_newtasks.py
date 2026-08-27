#!/usr/bin/env python3
"""G-Eval quality scoring for the 8 new tasks (task-expansion condition G).

Scores N_SAMPLE=30 responses per task per condition with the same GPT-4o
G-Eval judge (coherence/relevance/depth, 1-5) used for the original tasks:
conditions indep_t07 / indep_t10 / indep_t12 / prompt_v1.
Resumable per (condition, task). Output: results/task_expansion/quality_newtasks_*.json
"""
import sys
import os
import json
import glob
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from src.llm_agent import EXAMPLE_TASKS
from evaluate_quality_geval import evaluate_single

N_SAMPLE = 30
RETRIES = 6
NEW_TASKS = ['reasoning_2', 'factual_1', 'factual_2', 'debate_2',
             'ideation_1', 'ideation_2', 'ideation_3', 'creative_3']
CONDS = ['indep_t07', 'indep_t10', 'indep_t12', 'prompt_v1']


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
            time.sleep(delay)
            delay = min(delay * 2, 30)


def main():
    from openai import OpenAI
    client = OpenAI()
    prompts = {t.task_id: t.prompt for t in EXAMPLE_TASKS}
    rng = np.random.RandomState(0)

    out_dir = Path("results/task_expansion")
    old = sorted(glob.glob(str(out_dir / "quality_newtasks_2*.json")))
    out = json.load(open(old[-1])) if old else {
        "n_sample": N_SAMPLE, "timestamp": datetime.now().isoformat(), "scores": {}}
    of = out_dir / f"quality_newtasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    for tid in NEW_TASKS:
        f = sorted(glob.glob(f"results/task_expansion/pilot_{tid}_2*.json"))[-1]
        d = json.load(open(f))["independent"]
        for cond in CONDS:
            key = f"{cond}|{tid}"
            if len(out["scores"].get(key, [])) >= N_SAMPLE:
                print(f"[{key}] complete, skip", flush=True)
                continue
            texts = [r["text"] for r in d[cond]]
            idx = rng.choice(len(texts), N_SAMPLE, replace=False)
            print(f"[{key}] scoring {N_SAMPLE}...", flush=True)
            scores = list(out["scores"].get(key, []))
            for j, i in enumerate(idx[len(scores):], start=len(scores)):
                scores.append(score_retry(client, texts[int(i)], prompts[tid]))
                if (j + 1) % 10 == 0:
                    out["scores"][key] = scores
                    json.dump(out, open(of, "w"), indent=1)
            out["scores"][key] = scores
            json.dump(out, open(of, "w"), indent=1)
            print(f"    mean={np.mean(scores):.2f}", flush=True)

    json.dump(out, open(of, "w"), indent=1)
    print("\n=== means by condition (new tasks) ===")
    for cond in CONDS:
        vals = [np.mean(out["scores"][f"{cond}|{t}"]) for t in NEW_TASKS]
        print(f"{cond}: {np.mean(vals):.3f} (per-task {['%.2f' % v for v in vals]})")
    print(f"\nSaved: {of}\nDone: {datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()

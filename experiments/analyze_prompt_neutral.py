#!/usr/bin/env python3
"""Neutral-prompt control analysis (no API).

Computes the three geometric measures for the neutral-instruction condition
(politeness + two-paragraph format; no semantic direction) against the T=0.7
reference, on the paper's twelve tasks, with the paper's estimators. The
comparison of interest is its control-adjusted leakage against the
distinctiveness prompt's (0.29) and the held-out control (0 by construction).

Output: results/robustness/prompt_neutral_analysis.json
"""
import os, sys, json, glob
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_robustness_core import (load_gpt, leakage_block,
                                      participation_ratio, TASKS, SEED, OUT)
from analyze_12task_full import escape_block


def main():
    f = sorted(glob.glob("results/prompt_neutral/prompt_neutral_2*.json"))[-1]
    neu = json.load(open(f))["responses_by_task"]
    data = load_gpt()
    z = np.load(os.path.join(OUT, "emb_minilm.npz"))
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    rng = np.random.RandomState(SEED)

    per, acc = {}, {}
    for t in TASKS:
        texts = [x["text"] for x in neu[t]]
        E = np.asarray(model.encode(texts, show_progress_bar=False), np.float32)
        ref = z[f"ref07|{t}"]
        lk = leakage_block(ref, {"neutral": E, "prompt_v1": z[f"prompt_v1|{t}"]}, rng)
        esc, held = escape_block(ref, E, rng)
        mu = ref.mean(0)
        row = {"leak_neutral": round(lk["neutral"], 4),
               "leak_prompt_v1": round(lk["prompt_v1"], 4),
               "oor": round(esc, 4), "oor_held": round(held, 4),
               "deff": round(participation_ratio(E), 2),
               "deff_ref": round(participation_ratio(ref), 2),
               "radius_ratio": round(float(np.linalg.norm(E - mu, axis=1).mean()
                                     / np.linalg.norm(ref - mu, axis=1).mean()), 3)}
        per[t] = row
        for k, v in row.items():
            acc.setdefault(k, []).append(v)
        print(t, row, flush=True)

    means = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
    a, b = np.array(acc["leak_prompt_v1"]), np.array(acc["leak_neutral"])
    tt, p = stats.ttest_rel(a, b)
    res = {"config_file": f, "per_task": per, "task_means": means,
           "test_prompt_vs_neutral_leak": {
               "t11": round(float(tt), 2), "p": round(float(p), 5),
               "sign": f"{int((a > b).sum())}/{len(a)}"}}
    out = os.path.join(OUT, "prompt_neutral_analysis.json")
    json.dump(res, open(out, "w"), indent=1)
    print("\ntask means:", json.dumps(means, indent=1))
    print("prompt vs neutral leakage:", res["test_prompt_vs_neutral_leak"])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

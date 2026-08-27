#!/usr/bin/env python3
"""M3 check: do the extra dimensions at T=1.2 carry distinguishable content?

Applies the unique-idea counting method from the brainstorming case study
(run_brainstorming_case_study.count_unique_ideas, GPT-4o categorizer) to the
independent T=0.7 reference and the independent T=1.2 condition on the
original four tasks. For each task and condition we draw three disjoint
seeded subsamples of 40 responses and count distinct ideas per subsample,
giving 12 paired draws.

If T=1.2 adds only sampling noise, its unique-idea counts should not exceed
the reference's; if the added variance carries semantically distinguishable
content, they should.

Output: results/robustness/unique_ideas_temp.json
"""

import os, sys, json
import numpy as np
from scipy import stats
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_robustness_core import load_gpt, OUT
from run_brainstorming_case_study import CATEGORIZE_PROMPT


def count_unique_ideas(client, responses_texts, task_prompt, model="gpt-4o"):
    """Same categorizer as the case study, with json-mode output, a larger
    completion budget (40-response subsamples yield longer idea lists than the
    24-response pools it was written for), and a retry on malformed JSON."""
    all_text = "\n\n---\n\n".join(
        f"Response {i+1}:\n{t}" for i, t in enumerate(responses_texts))
    prompt = CATEGORIZE_PROMPT.format(task_prompt=task_prompt,
                                      all_responses=all_text)
    for attempt in range(3):
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=8000,
            response_format={"type": "json_object"})
        try:
            result = json.loads(completion.choices[0].message.content)
            return result["unique_ideas"], len(result["unique_ideas"])
        except (json.JSONDecodeError, KeyError) as e:
            if attempt == 2:
                raise
            print(f"    malformed JSON, retry {attempt+1}: {e}", flush=True)
from src.llm_agent import EXAMPLE_TASKS

TASKS4 = [t.task_id for t in EXAMPLE_TASKS[:4]]
PROMPTS = {t.task_id: t.prompt for t in EXAMPLE_TASKS[:4]}
DRAWS, N_SUB, SEED = 3, 40, 0


def main():
    client = OpenAI()
    data = load_gpt()
    rng = np.random.RandomState(SEED)
    res = {"config": {"draws": DRAWS, "n_sub": N_SUB, "judge": "gpt-4o",
                      "method": "run_brainstorming_case_study.count_unique_ideas"},
           "per_task": {}}
    ref_all, tmp_all = [], []
    for t in TASKS4:
        row = {"ref07": [], "temp12": []}
        for cond in ("ref07", "temp12"):
            texts = data[cond][t]
            idx = rng.permutation(len(texts))[:DRAWS * N_SUB]
            for d in range(DRAWS):
                sub = [texts[i] for i in idx[d * N_SUB:(d + 1) * N_SUB]]
                ideas, n = count_unique_ideas(client, sub, PROMPTS[t])
                row[cond].append(n)
                print(f"[{t}] {cond} draw {d+1}/{DRAWS}: {n} unique ideas", flush=True)
        res["per_task"][t] = row
        ref_all += row["ref07"]; tmp_all += row["temp12"]

    ref_all, tmp_all = np.array(ref_all, float), np.array(tmp_all, float)
    tt, p = stats.ttest_rel(tmp_all, ref_all)
    res["summary"] = {
        "ref_mean": round(float(ref_all.mean()), 2),
        "temp_mean": round(float(tmp_all.mean()), 2),
        "ratio": round(float(tmp_all.mean() / ref_all.mean()), 3),
        "paired_t": round(float(tt), 2), "df": len(ref_all) - 1,
        "p": round(float(p), 4),
        "sign": f"{int((tmp_all > ref_all).sum())}/{len(ref_all)}",
    }
    out = os.path.join(OUT, "unique_ideas_temp.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

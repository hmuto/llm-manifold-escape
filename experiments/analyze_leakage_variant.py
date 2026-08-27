#!/usr/bin/env python3
"""
Does the novelty finding survive a differently worded distinctiveness prompt?

Recompute subspace leakage (200 reference splits, k=20) with a SECOND prompt
condition: the differently worded distinctiveness instruction from
run_prompt_variant.py. If it leaks like the original prompt (well above temperature
and selection), the "prompt reaches the most new directions" finding is not specific
to the original phrasing.
"""

import sys, os, json, glob
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DYN = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
SPLITS, SEED, K = 200, 0, 20


def load_reference():
    f = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))[-1]
    return {t: [r["text"] for r in rs]
            for t, rs in json.load(open(f))["responses_by_task"].items()}


def load_dyn(cond):
    d = json.load(open(DYN)); pool = {}
    for td in d["conditions"][cond]:
        t = td["task_id"]; pool.setdefault(t, [])
        for tr in td["trials"]:
            for rt in tr.get("response_texts", []):
                pool[t].extend(r["text"] for r in rt)
    return pool


def load_temp12():
    f = sorted(glob.glob("results/temperature_expansion/temperature_expansion_2*.json"))[-1]
    d = json.load(open(f))["responses_by_temp_task"]["temp_1.2"]
    return {t: [r["text"] for r in tx] for t, tx in d.items()}


def load_json_task(pattern):
    f = sorted(glob.glob(pattern))[-1]
    return {t: [r["text"] for r in tx]
            for t, tx in json.load(open(f))["responses_by_task"].items()}


def captured(X, Vk):
    Xc = X - X.mean(0)
    return float(((Xc @ Vk) ** 2).sum() / (Xc ** 2).sum())


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda ts: np.asarray(model.encode(ts, show_progress_bar=False), dtype=float)
    conds = {"reference": load_reference(), "DDS (sel.)": load_dyn("dds_alpha_0.5"),
             "temperature T=1.2": load_temp12(),
             "prompt (original)": load_json_task("results/prompt_expansion/prompt_expansion_2*.json"),
             "prompt (variant)": load_json_task("results/prompt_variant/prompt_variant_2*.json")}
    tasks = list(conds["reference"].keys())
    E = {c: {t: enc(conds[c][t]) for t in tasks} for c in conds}
    LAB = ["DDS (sel.)", "temperature T=1.2", "prompt (original)", "prompt (variant)"]
    rng = np.random.RandomState(SEED)
    per_split = {c: [] for c in LAB}
    for _ in range(SPLITS):
        leak_t = {c: [] for c in LAB}
        for t in tasks:
            R = E["reference"][t]; n_test = len(R) // 2
            idx = rng.permutation(len(R))
            fit, held = R[idx[:len(R) - n_test]], R[idx[len(R) - n_test:]]
            Vk = PCA(n_components=K).fit(fit).components_.T
            ctrl = captured(held, Vk)
            for c in LAB:
                sub = E[c][t][rng.choice(len(E[c][t]), n_test, replace=False)]
                leak_t[c].append(ctrl - captured(sub, Vk))
        for c in LAB:
            per_split[c].append(float(np.mean(leak_t[c])))
    arr = {c: np.array(per_split[c]) for c in LAB}
    print(f"SPLITS={SPLITS}, k={K}\n")
    print(f"{'condition':<22} {'leakage [95% split interval]':>30}")
    out = {}
    for c in LAB:
        m, s = arr[c].mean(), arr[c].std(ddof=1)
        out[c] = {"mean": float(m), "sd": float(s)}
        print(f"{c:<22} {f'{m:.3f} [{m-1.96*s:.3f}, {m+1.96*s:.3f}]':>30}")
    both_gt_temp = np.mean((arr["prompt (variant)"] > arr["temperature T=1.2"]) &
                           (arr["prompt (original)"] > arr["temperature T=1.2"]))
    var_gt_sel = np.mean(arr["prompt (variant)"] > arr["DDS (sel.)"])
    print(f"\nvariant leaks > temperature in {np.mean(arr['prompt (variant)']>arr['temperature T=1.2'])*100:.0f}% of splits; "
          f"> selection in {var_gt_sel*100:.0f}%")
    print(f"both prompt wordings > temperature in {both_gt_temp*100:.0f}% of splits")
    Path("results/prompt_variant/leakage_variant.json").write_text(json.dumps(out, indent=2))
    print("Saved: results/prompt_variant/leakage_variant.json")


if __name__ == "__main__":
    main()

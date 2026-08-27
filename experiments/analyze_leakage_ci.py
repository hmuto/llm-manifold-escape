#!/usr/bin/env python3
"""
Uncertainty for the novelty (subspace-leakage) axis, to match the bootstrap CIs on
the effective-dimension axis.

For each of SPLITS random reference fit/held splits we compute, per condition, the
task-averaged captured fraction inside the reference top-k subspace and the leakage
(control - captured, control = held-out reference). The splits are PAIRED (the same
split drives every condition), so we report:
  - leakage mean +/- 1.96 x SD across splits (a stability interval), and
  - how often the ordering prompt > temperature > selection is preserved.
This shows the 0.20 / 0.10 / 0.05 ordering is not split noise. k = 20 (primary);
we also print k matched to the reference PC90 to show fixed-k is not doing the work.
"""

import sys, os, json, glob
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DYN = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
SPLITS, SEED = 200, 0


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


def load_prompt():
    f = sorted(glob.glob("results/prompt_expansion/prompt_expansion_2*.json"))[-1]
    return {t: [r["text"] for r in tx]
            for t, tx in json.load(open(f))["responses_by_task"].items()}


def captured(X, Vk):
    Xc = X - X.mean(0)
    return float(((Xc @ Vk) ** 2).sum() / (Xc ** 2).sum())


def pc90(E):
    ev = PCA(n_components=min(len(E), E.shape[1])).fit(E).explained_variance_ratio_
    return int(np.searchsorted(np.cumsum(ev), 0.90) + 1)


def run_k(E, tasks, K, label):
    rng = np.random.RandomState(SEED)
    LAB = ["DDS (sel.)", "prompt (distinct.)", "temperature T=1.2"]
    per_split = {c: [] for c in LAB}          # task-averaged leakage per split
    for _ in range(SPLITS):
        leak_t = {c: [] for c in LAB}
        for t in tasks:
            R = E["reference"][t]; n_test = len(R) // 2
            idx = rng.permutation(len(R))
            fit, held = R[idx[:len(R) - n_test]], R[idx[len(R) - n_test:]]
            k = K if isinstance(K, int) else max(2, min(pc90(fit), len(fit) - 1))
            Vk = PCA(n_components=k).fit(fit).components_.T
            ctrl = captured(held, Vk)
            for c in LAB:
                sub = E[c][t][rng.choice(len(E[c][t]), n_test, replace=False)]
                leak_t[c].append(ctrl - captured(sub, Vk))
        for c in LAB:
            per_split[c].append(float(np.mean(leak_t[c])))
    arr = {c: np.array(per_split[c]) for c in LAB}
    print(f"\n=== {label} ===")
    print(f"{'condition':<22} {'leakage [95% split interval]':>30}")
    for c in LAB:
        m, s = arr[c].mean(), arr[c].std(ddof=1)
        print(f"{c:<22} {f'{m:.3f} [{m-1.96*s:.3f}, {m+1.96*s:.3f}]':>30}")
    po = np.mean(arr["prompt (distinct.)"] > arr["temperature T=1.2"])
    ts = np.mean(arr["temperature T=1.2"] > arr["DDS (sel.)"])
    full = np.mean((arr["prompt (distinct.)"] > arr["temperature T=1.2"]) &
                   (arr["temperature T=1.2"] > arr["DDS (sel.)"]))
    print(f"ordering preserved across {SPLITS} splits: "
          f"prompt>temp {po*100:.0f}%, temp>selection {ts*100:.0f}%, full order {full*100:.0f}%")
    return {c: {"mean": float(arr[c].mean()), "sd": float(arr[c].std(ddof=1))} for c in LAB} | \
           {"order_prompt_gt_temp": float(po), "order_temp_gt_sel": float(ts), "order_full": float(full)}


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda ts: np.asarray(model.encode(ts, show_progress_bar=False), dtype=float)
    conds = {"reference": load_reference(), "DDS (sel.)": load_dyn("dds_alpha_0.5"),
             "prompt (distinct.)": load_prompt(), "temperature T=1.2": load_temp12()}
    tasks = list(conds["reference"].keys())
    E = {c: {t: enc(conds[c][t]) for t in tasks} for c in conds}
    print(f"SPLITS={SPLITS}; leakage = held-out-reference captured minus condition captured")
    out = {"k20": run_k(E, tasks, 20, "k = 20 (fixed)"),
           "pc90": run_k(E, tasks, "pc90", "k = reference PC90 (per split)")}
    Path("results/temperature_expansion/leakage_ci.json").write_text(json.dumps(out, indent=2))
    print("\nSaved: results/temperature_expansion/leakage_ci.json")


if __name__ == "__main__":
    main()

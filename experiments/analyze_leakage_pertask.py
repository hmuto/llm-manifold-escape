#!/usr/bin/env python3
"""
M1 follow-up: is the distinctiveness prompt's large subspace leakage driven by one
task, or is it consistent across tasks? (M3/M6 taught us not to trust 4-task
means.) Per-task captured-fraction inside the T=0.7 reference top-k subspace
(k=20), each set centred on its own mean; held-out reference = control.
"""

import sys, os, json, glob
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DYN = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
K = 20
SPLITS = 40
SEED = 0


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


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda ts: np.asarray(model.encode(ts, show_progress_bar=False), dtype=float)

    conds = {"reference": load_reference(), "DDS (sel.)": load_dyn("dds_alpha_0.5"),
             "prompt (distinct.)": load_prompt(), "temperature T=1.2": load_temp12()}
    tasks = list(conds["reference"].keys())
    E = {c: {t: enc(conds[c][t]) for t in tasks} for c in conds}
    rng = np.random.RandomState(SEED)
    print(f"per-task captured fraction inside reference top-{K} subspace "
          f"(leak vs held-out control in parens); {SPLITS} splits\n")
    print(f"{'task':<12} {'held-out':>10} {'DDS':>16} {'prompt':>16} {'temp T=1.2':>16}")
    out = {}
    for t in tasks:
        R = E["reference"][t]; n_test = len(R) // 2
        acc = {c: [] for c in ["held", "DDS (sel.)", "prompt (distinct.)", "temperature T=1.2"]}
        for _ in range(SPLITS):
            idx = rng.permutation(len(R))
            fit, held = R[idx[:len(R) - n_test]], R[idx[len(R) - n_test:]]
            Vk = PCA(n_components=K).fit(fit).components_.T
            acc["held"].append(captured(held, Vk))
            for c in ["DDS (sel.)", "prompt (distinct.)", "temperature T=1.2"]:
                X = E[c][t]; sub = X[rng.choice(len(X), n_test, replace=False)]
                acc[c].append(captured(sub, Vk))
        ctrl = np.mean(acc["held"])
        cells = {c: (np.mean(acc[c]), ctrl - np.mean(acc[c])) for c in acc}
        out[t] = {c: {"captured": float(cells[c][0]), "leak": float(cells[c][1])} for c in acc}

        def fmt(c):
            m, leak = cells[c]
            return f"{m:.3f} ({leak:+.3f})"
        print(f"{t:<12} {ctrl:>10.3f} {fmt('DDS (sel.)'):>16} "
              f"{fmt('prompt (distinct.)'):>16} {fmt('temperature T=1.2'):>16}")

    Path("results/temperature_expansion/leakage_pertask.json").write_text(json.dumps(out, indent=2))
    print("\nSaved: results/temperature_expansion/leakage_pertask.json")
    print("Read: positive leak = variance outside the reference span (new directions).")


if __name__ == "__main__":
    main()

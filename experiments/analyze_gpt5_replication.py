#!/usr/bin/env python3
"""Analyze the GPT-5 replication run with the paper's estimators.

Per task (original four), within-model against the GPT-5 ref07 reference:
  d_eff   participation ratio at native n (ref/temp/prompt 128, DDS pool 120)
  OOR     escape_block: cosine split-half, eps = 2x median NN, held-out control
  radius  mean Euclidean distance to the ref07 centroid, ratio to reference
  leakage leakage_block: 40 splits, k=20, held-out control adjustment

Output: results/gpt5_replication/gpt5_analysis.json (+ embedding cache npz)
"""

import os, sys, json, glob
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_robustness_core import leakage_block, participation_ratio, SEED
from analyze_12task_full import escape_block

OUT = "results/gpt5_replication"
CONDS = ["ref07", "temp12", "prompt_v1", "dds07"]


def load_texts():
    f = sorted(glob.glob(os.path.join(OUT, "gpt5_replication_2*.json")))[-1]
    d = json.load(open(f))
    tasks = d["config"]["tasks"]
    data = {}
    for t in tasks:
        sets = {c: [x["text"] for x in d["responses"][c][t]]
                for c in ("ref07", "temp12", "prompt_v1")}
        sets["dds07"] = [x["text"] for trial in d["dds"][t]
                         for rnd in trial for x in rnd]
        data[t] = sets
    print(f"loaded {f}")
    for t in tasks:
        print(" ", t, {c: len(v) for c, v in data[t].items()})
    return d, tasks, data


def embed_all(tasks, data):
    cache = os.path.join(OUT, "emb_gpt5.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    E = {}
    for t in tasks:
        for c in CONDS:
            E[f"{c}|{t}"] = np.asarray(
                model.encode(data[t][c], show_progress_bar=False), dtype=np.float32)
            print(f"embedded {c}|{t} n={len(E[f'{c}|{t}'])}", flush=True)
    np.savez_compressed(cache, **E)
    return E


def main():
    raw, tasks, data = load_texts()
    E = embed_all(tasks, data)
    rng = np.random.RandomState(SEED)

    per_task, acc = {}, {}
    for t in tasks:
        ref = E[f"ref07|{t}"]
        mu = ref.mean(0)
        ref_rad = float(np.linalg.norm(ref - mu, axis=1).mean())
        row = {"d_eff": {}, "oor": {}, "radius_ratio": {}, "leakage": {}}
        for c in CONDS:
            X = E[f"{c}|{t}"]
            row["d_eff"][c] = round(participation_ratio(X), 2)
            row["radius_ratio"][c] = round(
                float(np.linalg.norm(X - mu, axis=1).mean()) / ref_rad, 3)
        for c in ("dds07", "temp12", "prompt_v1"):
            esc, held = escape_block(ref, E[f"{c}|{t}"], rng)
            row["oor"][c] = round(esc, 4)
            row["oor"].setdefault("held_control", round(held, 4))
        lk = leakage_block(ref, {c: E[f"{c}|{t}"]
                                 for c in ("dds07", "temp12", "prompt_v1")}, rng)
        row["leakage"] = {c: round(v, 4) for c, v in lk.items()}
        per_task[t] = row
        for m in ("d_eff", "oor", "radius_ratio", "leakage"):
            for c, v in row[m].items():
                acc.setdefault((m, c), []).append(v)

    def tt(a, b, label):
        a, b = np.asarray(a, float), np.asarray(b, float)
        t_, p = stats.ttest_rel(a, b)
        return {"label": label, "t": round(float(t_), 2), "df": len(a) - 1,
                "p": round(float(p), 4),
                "sign": f"{int((a > b).sum())}/{len(a)}"}

    res = {
        "config": raw["config"],
        "per_task": per_task,
        "task_means": {f"{m}:{c}": round(float(np.mean(v)), 3)
                       for (m, c), v in acc.items()},
        "tests": [
            tt(acc[("d_eff", "temp12")], acc[("d_eff", "ref07")], "d_eff: temp vs ref"),
            tt(acc[("d_eff", "dds07")], acc[("d_eff", "ref07")], "d_eff: dds vs ref"),
            tt(acc[("d_eff", "temp12")], acc[("d_eff", "dds07")], "d_eff: temp vs dds"),
            tt(acc[("oor", "dds07")], acc[("oor", "held_control")], "OOR: dds vs held"),
            tt(acc[("leakage", "prompt_v1")], acc[("leakage", "dds07")], "leak: prompt vs dds"),
            tt(acc[("leakage", "prompt_v1")], acc[("leakage", "temp12")], "leak: prompt vs temp"),
        ],
    }
    out = os.path.join(OUT, "gpt5_analysis.json")
    json.dump(res, open(out, "w"), indent=1)
    print("\n=== task means ===")
    for k, v in sorted(res["task_means"].items()):
        print(f"  {k:24s} {v}")
    print("\n=== paired tests (t(3), 4 tasks) ===")
    for x in res["tests"]:
        print(f"  {x['label']:22s} t={x['t']:6.2f} p={x['p']:.4f} sign={x['sign']}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

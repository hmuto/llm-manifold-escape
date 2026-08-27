#!/usr/bin/env python3
"""
Bootstrap confidence intervals for the participation-ratio effective dimension.

The dimensional claim ("selection and the distinctiveness prompt do not add
directions; decoding does") rests on comparing effective dimensions across four
conditions. A referee will ask whether the differences survive sampling noise.
This script attaches a nonparametric bootstrap CI to each condition x task
effective dimension and reports a 4-condition x 4-task table.

Conditions (all embedded with all-MiniLM-L6-v2):
  reference    : independent N=128, temperature 0.7      (the fixed-decoding region)
  DDS          : alpha=0.5 selection pool, temperature 0.7 (selection lever)
  prompt       : distinctiveness instruction, temperature 0.7 (prompt lever)
  temperature  : independent N=128, temperature 1.2       (decoding lever)

For each condition x task:
  point estimate = participation ratio of the full embedded pool (reproduces the
                   numbers reported in the text).
  CI = point +/- 1.96 x SE, with SE the standard deviation of B nonparametric
       bootstrap resamples (drawn WITH replacement at the pool's native size).
       We use the bootstrap SE (a normal-approximation CI centred on the point
       estimate) rather than the naive percentile CI: with-replacement resampling
       creates ties that mildly deflate a variance-based statistic like the
       participation ratio, biasing percentile CIs downward off the point
       estimate; the bootstrap SE captures the sampling spread without that
       location bias.

Read: if the temperature CI sits ABOVE the reference/DDS/prompt CIs (no overlap),
decoding adds directions while selection and the prompt do not, beyond noise.
"""

import sys, os, json, glob
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DDS_FILE = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
B = 2000
SEED = 0


def load_ref_temp07():
    f = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))[-1]
    d = json.load(open(f))
    return {tid: [r["text"] for r in rs] for tid, rs in d["responses_by_task"].items()}


def load_dds_pool():
    d = json.load(open(DDS_FILE)); pool = {}
    for td in d["conditions"]["dds_alpha_0.5"]:
        tid = td["task_id"]; pool.setdefault(tid, [])
        for trial in td["trials"]:
            for rt in trial.get("response_texts", []):
                for resp in rt:
                    pool[tid].append(resp["text"])
    return pool


def load_prompt():
    f = sorted(glob.glob("results/prompt_expansion/prompt_expansion_2*.json"))[-1]
    d = json.load(open(f))
    return {tid: [r["text"] for r in tx]
            for tid, tx in d["responses_by_task"].items()}


def load_temp12():
    f = sorted(glob.glob("results/temperature_expansion/temperature_expansion_2*.json"))[-1]
    d = json.load(open(f))["responses_by_temp_task"]
    return {tid: [r["text"] for r in tx] for tid, tx in d["temp_1.2"].items()}


def participation_ratio(E):
    if len(E) < 3:
        return float("nan")
    ev = PCA(n_components=min(len(E), E.shape[1])).fit(E).explained_variance_
    return float((ev.sum() ** 2) / np.square(ev).sum())


def boot_se(E, rng):
    """Bootstrap SE at the pool's native size (resample with replacement)."""
    n = len(E)
    pr = np.array([participation_ratio(E[rng.randint(0, n, n)]) for _ in range(B)])
    return float(pr.std(ddof=1))


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda ts: np.asarray(model.encode(ts, show_progress_bar=False), dtype=float)

    conds = {"reference": load_ref_temp07(), "DDS": load_dds_pool(),
             "prompt": load_prompt(), "temperature": load_temp12()}
    tasks = list(conds["reference"].keys())

    E = {c: {t: enc(conds[c][t]) for t in tasks} for c in conds}
    print(f"point +/- 1.96 x bootstrap-SE (native n), B={B}\n")

    rng = np.random.RandomState(SEED)
    table = {c: {} for c in conds}
    print(f"{'task':<12} " + " ".join(f"{c:>22}" for c in conds))
    for t in tasks:
        cells = []
        for c in conds:
            pt = participation_ratio(E[c][t])
            se = boot_se(E[c][t], rng)
            table[c][t] = {"point": pt, "se": se, "ci": [pt - 1.96 * se, pt + 1.96 * se],
                           "n": len(E[c][t])}
            cells.append(f"{pt:5.1f} [{pt-1.96*se:4.1f},{pt+1.96*se:4.1f}]")
        print(f"{t:<12} " + " ".join(f"{x:>22}" for x in cells))

    # task-mean effective dimension, with SE on the mean via bootstrap
    print(f"\n{'MEAN':<12} " + " ".join(f"{c:>22}" for c in conds))
    means, meancells = {}, []
    for c in conds:
        pts = [table[c][t]["point"] for t in tasks]
        r2 = np.random.RandomState(SEED + 1)
        draws = np.array([np.mean([participation_ratio(
            E[c][t][r2.randint(0, len(E[c][t]), len(E[c][t]))]) for t in tasks])
            for _ in range(B)])
        se = float(draws.std(ddof=1)); pt = float(np.mean(pts))
        means[c] = {"point": pt, "se": se, "ci": [pt - 1.96 * se, pt + 1.96 * se]}
        meancells.append(f"{pt:5.1f} [{pt-1.96*se:4.1f},{pt+1.96*se:4.1f}]")
    print(f"{'':<12} " + " ".join(f"{x:>22}" for x in meancells))

    out = Path("results/temperature_expansion/effdim_bootstrap.json")
    json.dump({"B": B, "per_task": table, "mean": means}, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")

    # verdict
    ref, dds, prm, tmp = (means[c]["ci"] for c in ["reference", "DDS", "prompt", "temperature"])
    print("\nVERDICT (task-mean 95% CIs):")
    print(f"  reference   {means['reference']['point']:.1f} {ref}")
    print(f"  DDS         {means['DDS']['point']:.1f} {dds}")
    print(f"  prompt      {means['prompt']['point']:.1f} {prm}")
    print(f"  temperature {means['temperature']['point']:.1f} {tmp}")
    print(f"  temperature CI above reference CI (no overlap): {tmp[0] > ref[1]}")
    print(f"  DDS CI overlaps reference CI: {not (dds[1] < ref[0] or dds[0] > ref[1])}")
    print(f"  prompt CI overlaps reference CI: {not (prm[1] < ref[0] or prm[0] > ref[1])}")


if __name__ == "__main__":
    main()

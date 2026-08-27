#!/usr/bin/env python3
"""
Is the dimensional finding an artifact of the embedding model?

All d_eff results in the paper use all-MiniLM-L6-v2 (384-d). The whole dimensional
claim rests on the effective dimension, so we check that the KEY SEPARATION is not
embedding-specific: re-embed the same responses with a different model,
all-mpnet-base-v2 (768-d), and recompute the participation-ratio effective
dimension for the reference, the selection methods (DDS, MAP-Elites), and the
levers (temperature, distinctiveness prompt).

If, under the new embedding, selection still holds d_eff near the reference while
raising the decoding temperature still clears it, the discriminator (effective
dimension) is robust to the embedding choice.

point = participation ratio of the full embedded pool.
CI = point +/- 1.96 x bootstrap SE (native n, B resamples with replacement).
"""

import sys, os, json, glob
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DYN = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
EMB = "all-mpnet-base-v2"   # 768-d; different from the paper's all-MiniLM-L6-v2 (384-d)
B = 2000
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


def pr(E):
    if len(E) < 3:
        return float("nan")
    ev = PCA(n_components=min(len(E), E.shape[1])).fit(E).explained_variance_
    return float((ev.sum() ** 2) / np.square(ev).sum())


def se(E, rng):
    n = len(E)
    return float(np.array([pr(E[rng.randint(0, n, n)]) for _ in range(B)]).std(ddof=1))


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB)
    enc = lambda ts: np.asarray(model.encode(ts, show_progress_bar=False), dtype=float)

    conds = {"reference": load_reference(), "DDS (sel.)": load_dyn("dds_alpha_0.5"),
             "MAP-Elites": load_dyn("map_elites"),
             "temperature (T=1.2)": load_temp12(), "prompt (distinct.)": load_prompt()}
    tasks = list(conds["reference"].keys())
    E = {c: {t: enc(conds[c][t]) for t in tasks} for c in conds}
    dim = E["reference"][tasks[0]].shape[1]
    print(f"embedding = {EMB} ({dim}-d), B={B}\n")

    rng = np.random.RandomState(SEED)

    def mean_ci(Emap):
        r2 = np.random.RandomState(SEED + 1)
        draws = np.array([np.mean([pr(Emap[t][r2.randint(0, len(Emap[t]), len(Emap[t]))])
                                   for t in tasks]) for _ in range(B)])
        return float(np.mean([pr(Emap[t]) for t in tasks])), float(draws.std(ddof=1))

    ref_pt, ref_se = mean_ci(E["reference"])
    ref_lo, ref_hi = ref_pt - 1.96 * ref_se, ref_pt + 1.96 * ref_se
    print(f"{'condition':<22} {'d_eff [95% CI]':>20} {'ratio':>7}  verdict")
    print(f"{'reference (N=128)':<22} {f'{ref_pt:.1f} [{ref_lo:.1f},{ref_hi:.1f}]':>20} {'1.00x':>7}")
    out = {"embedding": EMB, "dim": dim, "reference": {"point": ref_pt, "se": ref_se}, "conds": {}}
    for c in conds:
        if c == "reference":
            continue
        pt, s = mean_ci(E[c]); lo, hi = pt - 1.96 * s, pt + 1.96 * s
        clears = lo > ref_hi
        overlaps = not (hi < ref_lo or lo > ref_hi)
        verdict = "CLEARS ref (adds dims)" if clears else ("overlaps ref (flat)" if overlaps else "below ref")
        out["conds"][c] = {"point": pt, "se": s, "ratio": pt / ref_pt, "clears_ref": clears}
        print(f"{c:<22} {f'{pt:.1f} [{lo:.1f},{hi:.1f}]':>20} {f'{pt/ref_pt:.2f}x':>7}  {verdict}")

    Path("results/temperature_expansion/effdim_embedding_robust.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: results/temperature_expansion/effdim_embedding_robust.json")
    print("Compare to all-MiniLM-L6-v2 (paper): DDS ~0.95x/flat, MAP-Elites ~1.0x/flat,")
    print("temperature 1.36x/CLEARS, prompt ~1.0x/flat. If the pattern holds under")
    print("all-mpnet (768-d), the dimensional discriminator is not embedding-specific.")


if __name__ == "__main__":
    main()

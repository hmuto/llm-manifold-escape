#!/usr/bin/env python3
"""
M1: does raising the temperature add variance in NEW directions, or just amplify
variance along the reference's existing axes?

The participation ratio rises whenever variance spreads more evenly over axes,
even with no new semantic direction, so "temperature -> d_eff up" could be a
near-mechanical property of higher sampling entropy. We test it directly at the
subspace level.

For each task we fit PCA on a FIT half of the T=0.7 reference and take its top-k
principal subspace V_k (the directions the reference actually uses). For a test
set X we center X on its OWN mean (so relocation / centroid shift is removed and
only the SHAPE of its spread is compared) and measure the fraction of X's variance
that lies INSIDE V_k:

    captured(X) = || X_c V_k ||^2 / || X_c ||^2 .

A held-out half of the reference gives the control: how much of a fresh
same-temperature sample the reference subspace captures. If temperature merely
amplifies existing axes, T=1.2 is captured as well as the held-out reference
(captured_temp ~ captured_ref). If temperature opens new directions, more of its
variance falls OUTSIDE V_k (captured_temp < captured_ref = leakage).

Selection (DDS) and the distinctiveness prompt are included so the test also
speaks to the central dichotomy: the claim predicts DDS stays inside the reference
subspace (captured ~ control) while temperature leaks out.

Secondary: the largest principal angle between V_k(reference-fit) and V_k(test),
against the fit-vs-heldout control; and the participation ratio of the LEAKED
(complement-projected) variance, to see whether leakage is a few new axes (low PR)
or isotropic noise (high PR).
"""

import sys, os, json, glob
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DYN = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
EMB = "all-MiniLM-L6-v2"
KS = [10, 20, 30]
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


def load_temp(tag):
    f = sorted(glob.glob("results/temperature_expansion/temperature_expansion_2*.json"))[-1]
    d = json.load(open(f))["responses_by_temp_task"][tag]
    return {t: [r["text"] for r in tx] for t, tx in d.items()}


def load_prompt():
    f = sorted(glob.glob("results/prompt_expansion/prompt_expansion_2*.json"))[-1]
    return {t: [r["text"] for r in tx]
            for t, tx in json.load(open(f))["responses_by_task"].items()}


def captured(X, Vk):
    Xc = X - X.mean(0)
    proj = Xc @ Vk
    return float((proj ** 2).sum() / (Xc ** 2).sum())


def resid_pr(X, Vk):
    """participation ratio of the variance OUTSIDE V_k (few new axes = low)."""
    Xc = X - X.mean(0)
    R = Xc - (Xc @ Vk) @ Vk.T
    ev = PCA(n_components=min(len(R) - 1, R.shape[1])).fit(R).explained_variance_
    return float((ev.sum() ** 2) / np.square(ev).sum())


def max_angle(Va, Vb):
    s = np.clip(np.linalg.svd(Va.T @ Vb, compute_uv=False), -1, 1)
    return float(np.degrees(np.arccos(s[-1])))


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB)
    enc = lambda ts: np.asarray(model.encode(ts, show_progress_bar=False), dtype=float)

    conds = {"reference": load_reference(), "DDS (sel.)": load_dyn("dds_alpha_0.5"),
             "prompt (distinct.)": load_prompt(),
             "temperature T=1.0": load_temp("temp_1.0"),
             "temperature T=1.2": load_temp("temp_1.2")}
    tasks = list(conds["reference"].keys())
    E = {c: {t: enc(conds[c][t]) for t in tasks} for c in conds}
    dim = E["reference"][tasks[0]].shape[1]
    print(f"embedding = {EMB} ({dim}-d); {SPLITS} random reference splits\n")
    print("captured = fraction of a set's own variance lying INSIDE the reference")
    print("top-k subspace (higher = stays in existing directions; lower = leaks to")
    print("new directions). 'held-out reference' is the same-temperature control.\n")

    rng = np.random.RandomState(SEED)
    out = {"embedding": EMB, "ks": KS, "results": {}}

    for k in KS:
        # per condition: accumulate captured over tasks x splits
        cap = {c: [] for c in ["held-out reference", "DDS (sel.)", "prompt (distinct.)",
                               "temperature T=1.0", "temperature T=1.2"]}
        ang = {c: [] for c in ["held-out reference", "temperature T=1.2"]}
        rpr = {"held-out reference": [], "temperature T=1.2": []}
        for t in tasks:
            R = E["reference"][t]
            n_test = len(R) // 2
            others = {c: E[c][t] for c in conds if c != "reference"}
            for _ in range(SPLITS):
                idx = rng.permutation(len(R))
                fit, held = R[idx[:len(R) - n_test]], R[idx[len(R) - n_test:]]
                Vk = PCA(n_components=k).fit(fit).components_.T          # (d,k)
                cap["held-out reference"].append(captured(held, Vk))
                for c, X in others.items():
                    sub = X[rng.choice(len(X), n_test, replace=False)]
                    cap[c].append(captured(sub, Vk))
                # secondary: principal angle & residual PR (control vs temperature)
                Vk_h = PCA(n_components=k).fit(held).components_.T
                ang["held-out reference"].append(max_angle(Vk, Vk_h))
                rpr["held-out reference"].append(resid_pr(held, Vk))
                Xt = others["temperature T=1.2"]
                subt = Xt[rng.choice(len(Xt), n_test, replace=False)]
                Vk_t = PCA(n_components=k).fit(subt).components_.T
                ang["temperature T=1.2"].append(max_angle(Vk, Vk_t))
                rpr["temperature T=1.2"].append(resid_pr(subt, Vk))

        print(f"===== reference top-k subspace, k = {k} (test n = half the reference) =====")
        ctrl = np.mean(cap["held-out reference"])
        print(f"{'condition':<22} {'captured':>9} {'leak vs ctrl':>13}")
        row = {}
        for c in ["held-out reference", "DDS (sel.)", "prompt (distinct.)",
                  "temperature T=1.0", "temperature T=1.2"]:
            m = float(np.mean(cap[c])); s = float(np.std(cap[c], ddof=1))
            leak = ctrl - m
            tag = "  <- control" if c == "held-out reference" else ""
            print(f"{c:<22} {m:8.3f} {leak:12.3f}{tag}")
            row[c] = {"captured": m, "sd": s, "leak_vs_control": leak}
        aH, aT = np.mean(ang["held-out reference"]), np.mean(ang["temperature T=1.2"])
        rH, rT = np.mean(rpr["held-out reference"]), np.mean(rpr["temperature T=1.2"])
        print(f"\n  max principal angle vs reference-fit:  held-out {aH:5.1f} deg   "
              f"T=1.2 {aT:5.1f} deg")
        print(f"  participation ratio of LEAKED variance: held-out {rH:5.1f}       "
              f"T=1.2 {rT:5.1f}   (low = few new axes)\n")
        out["results"][k] = {"captured": row, "control": ctrl,
                             "angle_heldout": float(aH), "angle_temp": float(aT),
                             "residpr_heldout": float(rH), "residpr_temp": float(rT)}

    Path("results/temperature_expansion/subspace_leakage.json").write_text(json.dumps(out, indent=2))
    print("Saved: results/temperature_expansion/subspace_leakage.json")
    print("\nRead: if temperature T=1.2 is captured much LESS than the held-out")
    print("reference (positive leak) while DDS is captured about as well as the")
    print("control, then temperature adds variance in directions OUTSIDE the")
    print("reference span (new directions), and selection does not -- the")
    print("relocation-vs-new-dimensions dichotomy is real, not a PR artifact.")


if __name__ == "__main__":
    main()

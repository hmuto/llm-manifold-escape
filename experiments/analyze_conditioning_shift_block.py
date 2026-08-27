#!/usr/bin/env python3
"""
M7: does the conditioning-shift MMD survive a permutation that respects trial blocks?

The original test (analyze_conditioning_shift.py) pools round-0 vs later-round
responses across all DDS trials and permutes INDIVIDUAL responses. But responses
within a trial share a lineage, so they are not exchangeable, and a response-level
permutation can be anti-conservative. Here we recompute the round-0-vs-later test
with a STRATIFIED (within-trial) permutation: within each trial the 24 responses
(8 round-0 + 16 later) are randomly relabelled 8 early / 16 late, then all trials
are pooled and the MMD is recomputed. This conditions on the trial, so the null
respects the block structure. We report the block p next to the original global-
permutation p for every task.
"""

import sys, os, json, glob
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DDS_FILE = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
COND = "dds_alpha_0.5"
N_PERM = 5000
SEED = 0


def load_by_trial():
    """Per task: list of trials, each = (round0_texts[8], later_texts[16])."""
    d = json.load(open(DDS_FILE)); out = {}
    for td in d["conditions"][COND]:
        tid = td["task_id"]; out.setdefault(tid, [])
        for trial in td["trials"]:
            rts = trial.get("response_texts", [])
            if not rts:
                continue
            r0 = [r["text"] for r in rts[0]]
            later = [r["text"] for rnd in rts[1:] for r in rnd]
            out[tid].append((r0, later))
    return out


def rbf_gram(X, Y, sigma):
    xx = (X * X).sum(1)[:, None]; yy = (Y * Y).sum(1)[None, :]
    d2 = xx + yy - 2.0 * X @ Y.T
    np.maximum(d2, 0, out=d2)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def mmd2_unbiased(X, Y, sigma):
    m, n = len(X), len(Y)
    Kxx = rbf_gram(X, X, sigma); Kyy = rbf_gram(Y, Y, sigma); Kxy = rbf_gram(X, Y, sigma)
    np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
    return (Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2.0 * Kxy.mean())


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def emb(texts):
        e = np.asarray(model.encode(texts, show_progress_bar=False), dtype=float)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    trials_by_task = load_by_trial()
    tasks = list(trials_by_task.keys())
    print(f"N_PERM={N_PERM}; within-trial (block) vs global permutation\n")
    print(f"{'task':<12} {'#trials':>7} {'MMD^2':>10} {'p_global':>9} {'p_block':>8}  verdict")
    out = {}
    for tid in tasks:
        trials = trials_by_task[tid]
        # embed each trial's 24 responses; first 8 = round-0, next 16 = later
        Es = [np.vstack([emb(r0), emb(lat)]) for (r0, lat) in trials]
        sizes = [(len(r0), len(lat)) for (r0, lat) in trials]
        early = np.vstack([E[:s0] for E, (s0, _) in zip(Es, sizes)])
        late = np.vstack([E[s0:] for E, (s0, _) in zip(Es, sizes)])
        P = np.vstack([early, late])
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        sigma = float(np.sqrt(np.median(d2[d2 > 0]) / 2.0))
        obs = mmd2_unbiased(early, late, sigma)

        rng = np.random.RandomState(SEED)
        # global permutation (original): shuffle all labels ignoring trials
        m = len(early); Z = np.vstack([early, late]); cg = 0
        for _ in range(N_PERM):
            idx = rng.permutation(len(Z))
            if mmd2_unbiased(Z[idx[:m]], Z[idx[m:]], sigma) >= obs:
                cg += 1
        p_global = (cg + 1) / (N_PERM + 1)

        # block permutation: within each trial relabel s0 early / s1 late
        cb = 0
        for _ in range(N_PERM):
            e_list, l_list = [], []
            for E, (s0, s1) in zip(Es, sizes):
                perm = rng.permutation(len(E))
                e_list.append(E[perm[:s0]]); l_list.append(E[perm[s0:]])
            if mmd2_unbiased(np.vstack(e_list), np.vstack(l_list), sigma) >= obs:
                cb += 1
        p_block = (cb + 1) / (N_PERM + 1)

        v = "sig (block)" if p_block < 0.05 else "NOT sig (block)"
        out[tid] = {"n_trials": len(trials), "mmd2": obs, "sigma": sigma,
                    "p_global": p_global, "p_block": p_block}
        print(f"{tid:<12} {len(trials):>7} {obs:>10.5f} {p_global:>9.4f} {p_block:>8.4f}  {v}")

    sig_b = sum(1 for v in out.values() if v["p_block"] < 0.05)
    sig_g = sum(1 for v in out.values() if v["p_global"] < 0.05)
    print(f"\nsignificant (p<0.05): global {sig_g}/{len(tasks)}, block {sig_b}/{len(tasks)}")
    Path("results/support_vs_loop").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("results/support_vs_loop/conditioning_shift_block.json", "w"), indent=2)
    print("Saved: results/support_vs_loop/conditioning_shift_block.json")
    print("Read: if the block p stays < 0.05, the loop-moves-the-distribution claim")
    print("survives a permutation that respects trial exchangeability (M7 addressed).")


if __name__ == "__main__":
    main()

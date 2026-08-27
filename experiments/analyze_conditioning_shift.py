#!/usr/bin/env python3
"""
Does the regeneration loop move the distribution (conditioning), even though it
does not escape the support?

The regeneration loop conditions each round's generation on the surviving
responses, so later-round responses are drawn from a CONDITIONAL distribution
that could, in principle, differ from the single-shot UNCONDITIONAL distribution
(round 0). We test directly whether it does, using a two-sample kernel test
(MMD) between:
  - unconditional set U = round-0 responses (generated with no context), and
  - conditional set   C = later-round responses (rounds 1-2, regenerated on the
    loop's context),
pooled per task across DDS (alpha=0.5) trials.

A significant MMD means the loop genuinely MOVES the response distribution. Read
together with the support analysis (matched-k: no escape) and the novel-region
fraction, this gives the non-trivial statement: the loop moves the distribution
along the manifold, yet the moved distribution stays within the single-shot
support -- selection plus conditional regeneration, an information channel that
could have escaped, does not.

MMD reference: Gretton et al., JMLR 2012 (unbiased estimator, RBF kernel with the
median heuristic, permutation test).
"""

import json, glob
import numpy as np
from pathlib import Path

DDS_FILE = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
N_PERM = 2000
COND = "dds_alpha_0.5"


def load_by_round():
    """Per task: round-0 texts (unconditional) and later-round texts (conditional)."""
    d = json.load(open(DDS_FILE))
    r0, later = {}, {}
    for td in d["conditions"][COND]:
        tid = td["task_id"]; r0.setdefault(tid, []); later.setdefault(tid, [])
        for trial in td["trials"]:
            rts = trial.get("response_texts", [])
            for ri, round_texts in enumerate(rts):
                for resp in round_texts:
                    (r0 if ri == 0 else later)[tid].append(resp["text"])
    return r0, later


def load_independent():
    f = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))[-1]
    d = json.load(open(f))
    return {tid: [r["text"] for r in rs] for tid, rs in d["responses_by_task"].items()}


def rbf_gram(X, Y, sigma):
    # squared euclidean distances
    xx = (X * X).sum(1)[:, None]; yy = (Y * Y).sum(1)[None, :]
    d2 = xx + yy - 2.0 * X @ Y.T
    np.maximum(d2, 0, out=d2)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def mmd2_unbiased(X, Y, sigma):
    m, n = len(X), len(Y)
    Kxx = rbf_gram(X, X, sigma); Kyy = rbf_gram(Y, Y, sigma); Kxy = rbf_gram(X, Y, sigma)
    np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
    return (Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1))
            - 2.0 * Kxy.mean())


def perm_test(X, Y, sigma, n_perm=N_PERM, seed=0):
    obs = mmd2_unbiased(X, Y, sigma)
    Z = np.vstack([X, Y]); m = len(X); rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(Z))
        if mmd2_unbiased(Z[idx[:m]], Z[idx[m:]], sigma) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    r0, later = load_by_round()
    indep = load_independent()
    tasks = list(r0.keys())

    def emb(texts):
        e = np.asarray(model.encode(texts, show_progress_bar=False), dtype=float)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)   # unit-norm

    print("=" * 70)
    print("CONDITIONING SHIFT: round-0 (unconditional) vs later rounds (conditional)")
    print("=" * 70)
    print(f"{'task':<12} {'n0':>4} {'nL':>4} {'MMD^2(r0,later)':>16} {'p':>8}   "
          f"{'MMD^2(indep,later)':>18} {'p':>8}")
    out = {}
    for tid in tasks:
        E0, EL, EI = emb(r0[tid]), emb(later[tid]), emb(indep[tid])
        # median heuristic bandwidth on the pooled (r0+later) sample
        P = np.vstack([E0, EL])
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        sigma = float(np.sqrt(np.median(d2[d2 > 0]) / 2.0))
        mmd_c, p_c = perm_test(E0, EL, sigma, seed=0)
        mmd_i, p_i = perm_test(EI, EL, sigma, seed=1)
        out[tid] = {"n_r0": len(E0), "n_later": len(EL), "sigma": sigma,
                    "mmd2_r0_later": mmd_c, "p_r0_later": p_c,
                    "mmd2_indep_later": mmd_i, "p_indep_later": p_i}
        print(f"{tid:<12} {len(E0):>4} {len(EL):>4} {mmd_c:>16.5f} {p_c:>8.4f}   "
              f"{mmd_i:>18.5f} {p_i:>8.4f}")

    Path("results/support_vs_loop").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("results/support_vs_loop/conditioning_shift.json", "w"), indent=2)
    sig = sum(1 for v in out.values() if v["p_r0_later"] < 0.05)
    print(f"\nround-0 vs later significant (p<0.05): {sig}/{len(tasks)} tasks")
    print("Interpretation: significant MMD => the loop moves the response distribution")
    print("(conditional != unconditional); combined with matched-k (no escape) and the")
    print("novel-region fraction, the loop moves the distribution but stays in the support.")
    print("\nSaved: results/support_vs_loop/conditioning_shift.json")


if __name__ == "__main__":
    main()

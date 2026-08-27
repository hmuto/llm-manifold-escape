#!/usr/bin/env python3
"""
Is the dimensional finding an artifact of the LINEAR measure?

Every d_eff in the paper is the participation ratio, a LINEAR (PCA) measure of how
many directions carry variance. The title speaks of the "manifold" and of "new
dimensions", which are intrinsic (nonlinear) notions. A linear measure can rise
from nonlinear scatter without a genuinely new axis, and can miss a new curved
direction. So we test the SAME ordering with two standard INTRINSIC-dimension
estimators that make no linearity assumption:

  TwoNN (Facco, d'Errico, Rodriguez, Laio, Sci. Rep. 2017): parameter-free; uses
    the ratio of the 2nd- to 1st-nearest-neighbour distance per point.
  MLE  (Levina & Bickel, NIPS 2004): maximum-likelihood local estimate averaged
    over a range of k nearest neighbours.

If, under the intrinsic estimators, selection (DDS, MAP-Elites) and the
distinctiveness prompt stay near the independent reference while raising the
decoding temperature rises above it -- the same ordering the linear d_eff shows --
then "reaches the tails but not new dimensions" is not an artifact of using a
linear measure, and the title's manifold/dimension language is supported.

Intrinsic-dimension estimators are downward-biased at finite n and high true
dimension, and their absolute values differ from the linear participation ratio;
the informative quantity is the CROSS-CONDITION ordering at MATCHED n, not the
absolute number. We therefore subsample every condition to a common per-task n and
report a subsampling interval (point at full matched n; SE from m<n subsamples
without replacement, which avoids the nearest-neighbour ties that a with-
replacement bootstrap would create).
"""

import sys, os, json, glob
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DYN = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
EMB = "all-MiniLM-L6-v2"     # the paper's PRIMARY embedding (matches the d_eff tables)
B = 500
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


def twonn(X, discard=0.1):
    """Intrinsic dimension by the TwoNN estimator (slope through the origin of
    -log(1-F(mu)) vs log(mu), mu = r2/r1, discarding the top-`discard` tail)."""
    n = len(X)
    if n < 10:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dist, _ = nn.kneighbors(X)          # col0 = self (0), col1 = r1, col2 = r2
    r1, r2 = dist[:, 1], dist[:, 2]
    m = r1 > 0
    mu = r2[m] / r1[m]
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1)])
    N = len(mu)
    F = np.arange(1, N + 1) / N
    keep = int(N * (1 - discard))
    x = np.log(mu[:keep]); y = -np.log(1.0 - F[:keep])
    return float(np.sum(x * y) / np.sum(x * x))


def mle_id(X, k1=5, k2=15):
    """Levina-Bickel maximum-likelihood intrinsic dimension, averaged over k."""
    n = len(X)
    if n <= k2 + 1:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k2 + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    dist = dist[:, 1:]                   # drop self
    ests = []
    for k in range(k1, k2 + 1):
        Tk = dist[:, k - 1][:, None]     # distance to k-th neighbour
        prev = dist[:, :k - 1]           # distances to 1..k-1
        good = np.all(prev > 0, axis=1) & (Tk[:, 0] > 0)
        logs = np.log(Tk[good] / prev[good])
        mk = (k - 2) / logs.sum(axis=1)  # (k-1)-1 = k-2 in the ML correction
        ests.append(np.mean(mk))
    return float(np.mean(ests))


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

    # matched n per task = min pool over conditions (fair finite-sample comparison)
    nmatch = {t: min(len(E[c][t]) for c in conds) for t in tasks}
    r0 = np.random.RandomState(SEED)
    Em = {c: {t: E[c][t][r0.choice(len(E[c][t]), nmatch[t], replace=False)]
              for t in tasks} for c in conds}
    print(f"embedding = {EMB} ({dim}-d); matched n/task = "
          f"{ {t: nmatch[t] for t in tasks} }; B={B} subsamples\n")

    def mean_ci(estfn, Emap):
        rng = np.random.RandomState(SEED + 1)
        point = float(np.mean([estfn(Emap[t]) for t in tasks]))
        draws = []
        for _ in range(B):
            vals = []
            for t in tasks:
                X = Emap[t]; m = int(0.85 * len(X))
                vals.append(estfn(X[rng.choice(len(X), m, replace=False)]))
            draws.append(np.mean(vals))
        return point, float(np.std(draws, ddof=1))

    for name, estfn in [("TwoNN", twonn), ("MLE (k=5..15)", mle_id)]:
        print(f"===== intrinsic dimension: {name} =====")
        ref_pt, ref_se = mean_ci(estfn, Em["reference"])
        ref_lo, ref_hi = ref_pt - 1.96 * ref_se, ref_pt + 1.96 * ref_se
        print(f"{'condition':<22} {'ID [95% CI]':>18} {'ratio':>7}  verdict")
        print(f"{'reference':<22} {f'{ref_pt:.2f} [{ref_lo:.2f},{ref_hi:.2f}]':>18} {'1.00x':>7}")
        block = {"estimator": name, "reference": {"point": ref_pt, "se": ref_se}, "conds": {}}
        for c in conds:
            if c == "reference":
                continue
            pt, s = mean_ci(estfn, Em[c]); lo, hi = pt - 1.96 * s, pt + 1.96 * s
            clears = lo > ref_hi
            verdict = ("CLEARS ref (adds dims)" if clears else
                       ("overlaps ref (flat)" if not (hi < ref_lo) else "below ref"))
            block["conds"][c] = {"point": pt, "se": s, "ratio": pt / ref_pt, "clears_ref": clears}
            print(f"{c:<22} {f'{pt:.2f} [{lo:.2f},{hi:.2f}]':>18} {f'{pt/ref_pt:.2f}x':>7}  {verdict}")
        print()
        out = Path(f"results/temperature_expansion/intrinsic_dim_{name.split()[0].lower()}.json")
        out.write_text(json.dumps(block, indent=2))
        print(f"Saved: {out}\n")

    print("Read: if temperature CLEARS the reference while DDS/MAP-Elites/prompt")
    print("overlap it -- the SAME ordering as the linear d_eff -- then 'reaches the")
    print("tails but not new dimensions' is not an artifact of the linear measure.")


if __name__ == "__main__":
    main()

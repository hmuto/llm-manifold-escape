#!/usr/bin/env python3
"""Pilot sanity analysis for the task-expansion run (paper/task_expansion_plan.md §7-2).

For each pilot task, computes the paper's three-form measures and checks they land
in plausible ranges:
  - d_eff (participation ratio, matched n) for indep T=0.7/1.0/1.2, prompt v1/v2,
    DDS pools (T=0.7 / T=1.2), Debate, MAP-Elites
  - escape + radius vs the independent T=0.7 reference (split-half, eps = 2x median
    NN distance, 50 splits) for DDS pool / T=1.2 / prompt
  - leakage vs the reference top-k principal subspace (k=10/20/30, own-centred,
    held-out reference control)

Usage: python3 analyze_task_expansion_pilot.py [task_id ...]
       (default: factual_1 ideation_1)
"""
import sys
import os
import glob
import json
import numpy as np

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

MODEL = SentenceTransformer('all-MiniLM-L6-v2')
RNG = np.random.default_rng(0)


def embed(texts):
    return np.asarray(MODEL.encode(texts, show_progress_bar=False))


def d_eff(X):
    ev = PCA(n_components=min(len(X) - 1, X.shape[1])).fit(X).explained_variance_
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def d_eff_matched(X, n, reps=20):
    """Participation ratio at matched sample size n (mean over subsamples)."""
    if len(X) <= n:
        return d_eff(X)
    return float(np.mean([d_eff(X[RNG.choice(len(X), n, replace=False)])
                          for _ in range(reps)]))


def escape_radius(ref, test, n_splits=50):
    """Split-half escape (eps = 2x median NN dist in the reference half) and
    radius ratio vs the full reference centroid. Matched sample sizes."""
    n = len(ref) // 2
    esc_test, esc_held = [], []
    for _ in range(n_splits):
        idx = RNG.permutation(len(ref))
        R, H = ref[idx[:n]], ref[idx[n:2 * n]]
        d_RR = np.linalg.norm(R[:, None] - R[None], axis=-1)
        np.fill_diagonal(d_RR, np.inf)
        eps = 2 * np.median(d_RR.min(1))
        Tm = test[RNG.choice(len(test), min(n, len(test)), replace=False)]
        d_TR = np.linalg.norm(Tm[:, None] - R[None], axis=-1).min(1)
        d_HR = np.linalg.norm(H[:, None] - R[None], axis=-1).min(1)
        esc_test.append((d_TR > eps).mean())
        esc_held.append((d_HR > eps).mean())
    c = ref.mean(0)
    radius_ratio = np.linalg.norm(test - c, axis=1).mean() / \
        np.linalg.norm(ref - c, axis=1).mean()
    return float(np.mean(esc_test)), float(np.mean(esc_held)), float(radius_ratio)


def leakage(ref, test, ks=(10, 20, 30), n_splits=50):
    """Own-centred variance outside the reference top-k subspace, minus the
    held-out reference control (mean over ks and splits)."""
    vals = []
    n = len(ref) // 2
    for _ in range(n_splits):
        idx = RNG.permutation(len(ref))
        F, H = ref[idx[:n]], ref[idx[n:2 * n]]
        for k in ks:
            Vk = PCA(n_components=k).fit(F).components_.T          # (d, k)

            def outside(X):
                Xc = X - X.mean(0)
                tot = (Xc ** 2).sum()
                inside = ((Xc @ Vk) ** 2).sum()
                return 1.0 - inside / tot

            vals.append(outside(test) - outside(H))
    return float(np.mean(vals))


def pool_loop_texts(trials):
    return [r['text'] for tr in trials for rt in tr['response_texts'] for r in rt]


def main():
    tasks = sys.argv[1:] or ['factual_1', 'ideation_1']
    out = {}
    for tid in tasks:
        fs = sorted(glob.glob(f'results/task_expansion/pilot_{tid}_2*.json'))
        if not fs:
            print(f'[{tid}] no result file yet'); continue
        d = json.load(open(fs[-1]))
        ind = d['independent']
        missing = [k for k in ('indep_t07', 'indep_t10', 'indep_t12',
                               'prompt_v1', 'prompt_v2') if len(ind.get(k, [])) < 2]
        if missing:
            print(f'[{tid}] incomplete independent conditions: {missing}')
        E = {k: embed([r['text'] for r in v]) for k, v in ind.items() if len(v) >= 2}
        L = {k: embed(pool_loop_texts(v)) for k, v in d['loops'].items() if v}

        ref = E.get('indep_t07')
        r = {'n': {k: len(v) for k, v in list(E.items()) + list(L.items())}}

        # matched-n d_eff (n = smallest pool, typically 120)
        nm = min(len(v) for v in list(E.values()) + list(L.values()))
        r['d_eff_matched_n'] = nm
        r['d_eff'] = {k: round(d_eff_matched(v, nm), 1)
                      for k, v in list(E.items()) + list(L.items())}

        if ref is not None:
            r['escape'] = {}
            for k in ('dds_a05_t07', 'indep_t12', 'prompt_v1'):
                X = L.get(k) if k in L else E.get(k)
                if X is None:
                    continue
                et, eh, rad = escape_radius(ref, X)
                r['escape'][k] = {'escape': round(et, 3), 'heldout_ctrl': round(eh, 3),
                                  'radius_ratio': round(rad, 2)}
            r['leakage'] = {}
            for k in ('dds_a05_t07', 'indep_t12', 'prompt_v1', 'prompt_v2'):
                X = L.get(k) if k in L else E.get(k)
                if X is None:
                    continue
                r['leakage'][k] = round(leakage(ref, X), 3)
        out[tid] = r

        print(f'\n===== {tid} =====')
        print(' n per condition:', r['n'])
        print(f" d_eff (matched n={r['d_eff_matched_n']}):")
        for k, v in r['d_eff'].items():
            print(f'   {k:16s} {v}')
        if 'escape' in r:
            print(' escape / radius vs indep_t07:')
            for k, v in r['escape'].items():
                print(f"   {k:16s} esc={v['escape']:.3f} (ctrl {v['heldout_ctrl']:.3f}) "
                      f"radius={v['radius_ratio']:.2f}x")
            print(' leakage (above held-out control):')
            for k, v in r['leakage'].items():
                print(f'   {k:16s} {v:+.3f}')

    json.dump(out, open('results/task_expansion/pilot_analysis.json', 'w'), indent=1)
    print('\nsaved -> results/task_expansion/pilot_analysis.json')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Estimator-threshold sensitivity for the two headline measures.

(1) Out-of-reference rate: the paper's epsilon is 2x the median
nearest-neighbour distance inside the reference. Here the multiplier is
swept over {1.5, 2.0, 2.5, 3.0} for the DDS pool and the held-out control
on all twelve tasks, with the paper's split-half protocol.

(2) Subspace leakage: the paper fixes k=20 reference components. Here k is
chosen per task as the number of components needed to reach 90% explained
variance on the fit half, and leakage is recomputed for DDS, temperature,
and the distinctiveness prompt.

Everything runs from the cached MiniLM embeddings; no API calls.
Output: results/robustness/threshold_sensitivity.json
"""

import os, sys, json
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_robustness_core import TASKS, SEED, OUT

MULTS = (1.5, 2.0, 2.5, 3.0)
SPLITS = 50
EVR = 0.90


def escape_mult(ref, test, rng, mult, n_splits=SPLITS):
    n = len(ref) // 2
    esc_t, esc_h = [], []
    for _ in range(n_splits):
        idx = rng.permutation(len(ref))
        R, H = ref[idx[:n]], ref[idx[n:2 * n]]
        rd = cosine_distances(R)
        np.fill_diagonal(rd, np.inf)
        eps = float(np.median(rd.min(axis=1))) * mult
        Tm = test[rng.choice(len(test), min(n, len(test)), replace=False)]
        esc_t.append(float(np.mean(cosine_distances(Tm, R).min(axis=1) > eps)))
        esc_h.append(float(np.mean(cosine_distances(H, R).min(axis=1) > eps)))
    return float(np.mean(esc_t)), float(np.mean(esc_h))


def captured(X, Vk):
    Xc = X - X.mean(0)
    return float(((Xc @ Vk) ** 2).sum()) / float((Xc ** 2).sum())


def leakage_evr(ref, sets, rng, n_splits=40):
    n = len(ref) // 2
    acc = {c: [] for c in sets}
    acc['held'] = []
    kk = []
    for _ in range(n_splits):
        idx = rng.permutation(len(ref))
        fit, held = ref[idx[:n]], ref[idx[n:2 * n]]
        pca = PCA().fit(fit)
        k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_),
                                EVR) + 1)
        kk.append(k)
        Vk = pca.components_[:k].T
        acc['held'].append(captured(held, Vk))
        for c, X in sets.items():
            sub = X[rng.choice(len(X), min(n, len(X)), replace=False)]
            acc[c].append(captured(sub, Vk))
    held = float(np.mean(acc['held']))
    out = {c: round(held - float(np.mean(acc[c])), 4) for c in sets}
    out['k_mean'] = round(float(np.mean(kk)), 1)
    return out


def main():
    z = np.load(os.path.join(OUT, 'emb_minilm.npz'))
    rng = np.random.RandomState(SEED)
    res = {'oor_mult': {}, 'leakage_evr': {}, 'config': {
        'mults': MULTS, 'evr': EVR, 'splits': SPLITS}}

    means = {m: {'dds': [], 'held': []} for m in MULTS}
    for t in TASKS:
        ref, dds = z[f'ref07|{t}'], z[f'dds07|{t}']
        row = {}
        for m in MULTS:
            esc, held = escape_mult(ref, dds, rng, m)
            row[str(m)] = {'dds': round(esc, 4), 'held': round(held, 4)}
            means[m]['dds'].append(esc)
            means[m]['held'].append(held)
        res['oor_mult'][t] = row
        print(f'[{t}] oor by mult:',
              {m: row[str(m)]['dds'] for m in MULTS}, flush=True)
    res['oor_mult_means'] = {
        str(m): {'dds': round(float(np.mean(v['dds'])), 4),
                 'held': round(float(np.mean(v['held'])), 4),
                 'dds_minus_held': round(float(np.mean(v['dds'])) -
                                         float(np.mean(v['held'])), 4)}
        for m, v in means.items()}

    lacc = {}
    for t in TASKS:
        ref = z[f'ref07|{t}']
        sets = {'dds': z[f'dds07|{t}'], 'temp': z[f'temp12|{t}'],
                'prompt': z[f'prompt_v1|{t}']}
        out = leakage_evr(ref, sets, rng)
        res['leakage_evr'][t] = out
        for c in ('dds', 'temp', 'prompt', 'k_mean'):
            lacc.setdefault(c, []).append(out[c])
        print(f'[{t}] evr-k leakage:', out, flush=True)
    res['leakage_evr_means'] = {c: round(float(np.mean(v)), 4)
                                for c, v in lacc.items()}
    ordering = sum(1 for t in TASKS
                   if res['leakage_evr'][t]['prompt'] >
                   max(res['leakage_evr'][t]['dds'],
                       res['leakage_evr'][t]['temp']))
    res['leakage_evr_means']['prompt_highest_tasks'] = f'{ordering}/12'

    out = os.path.join(OUT, 'threshold_sensitivity.json')
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res['oor_mult_means'], indent=1))
    print(json.dumps(res['leakage_evr_means'], indent=1))
    print('Saved:', out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Consolidated 12-task analysis (task-expansion full run).

Uses the PAPER'S exact estimators:
  - d_eff: participation ratio (PCA explained variance), bootstrap CI
    (B=2000, resample with replacement at native n, point +/- 1.96*SE)
    [analyze_effdim_bootstrap.py]
  - escape: split-half of independent-128, eps = 2x median NN COSINE distance
    within the reference half; escape = fraction of a matched-size subset with
    min cosine distance to R > eps; held-out independent half = control;
    50 splits [analyze_dds_escape.py]
  - leakage: reference T=0.7 split fit/held, top-K=20 PCA subspace from fit,
    captured = own-centred variance fraction inside; leak = captured(held) -
    captured(set); 40 splits [analyze_leakage_pertask.py]

Old 4 tasks read from archived files; new 8 from results/task_expansion/.
Task-level inference (n=12): paired t + Wilcoxon on per-task values.
Openness gradient: pre-registered ranks reasoning<factual<debate<ideation<creative.
Output: results/task_expansion/full12_analysis.json + printed tables.
"""
import os
import sys
import json
import glob
import numpy as np
from scipy import stats

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

B = 2000
ESC_SPLITS = 50
LEAK_SPLITS = 40
LEAK_K = 20
SEED = 0

OLD_TASKS = ['creative_1', 'creative_2', 'problem_1', 'debate_1']
NEW_TASKS = ['reasoning_2', 'factual_1', 'factual_2', 'debate_2',
             'ideation_1', 'ideation_2', 'ideation_3', 'creative_3']
TASKS = OLD_TASKS + NEW_TASKS

# Pre-registered openness ranks (task_expansion_plan.md §5-3).
OPENNESS = {'problem_1': 1, 'reasoning_2': 1, 'factual_1': 2, 'factual_2': 2,
            'debate_1': 3, 'debate_2': 3, 'ideation_1': 4, 'ideation_2': 4,
            'ideation_3': 4, 'creative_1': 5, 'creative_2': 5, 'creative_3': 5}
# Alternate classification: creative_2 (ocean-plastic solution) read as ideation.
OPENNESS_ALT = dict(OPENNESS, creative_2=4)

CONDS = ['ref07', 'temp10', 'temp12', 'prompt_v1', 'prompt_v2',
         'dds07', 'dds12', 'debate', 'map']


def texts(v):
    return [r['text'] if isinstance(r, dict) else r for r in v]


def pool_trials(cond_block, tid):
    for td in cond_block:
        if td['task_id'] == tid:
            return [r['text'] for tr in td['trials']
                    for rt in tr['response_texts'] for r in rt]
    return []


def load_all():
    data = {c: {} for c in CONDS}
    # ---- old tasks
    ref = json.load(open(sorted(glob.glob(
        'results/independent_scaling/independent_scaling_*.json'))[-1]))['responses_by_task']
    tmp = json.load(open(sorted(glob.glob(
        'results/temperature_expansion/temperature_expansion_2*.json'))[-1]))['responses_by_temp_task']
    pv1 = json.load(open(sorted(glob.glob(
        'results/prompt_expansion/prompt_expansion_2*.json'))[-1]))['responses_by_task']
    pv2 = json.load(open(sorted(glob.glob(
        'results/prompt_variant/prompt_variant_2*.json'))[-1]))['responses_by_task']
    dyn = json.load(open(
        'results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json'))['conditions']
    dds12 = json.load(open(sorted(glob.glob(
        'results/temperature_expansion/dds_temp12_*.json'))[-1]))['conditions']
    for t in OLD_TASKS:
        data['ref07'][t] = texts(ref[t])
        data['temp10'][t] = texts(tmp['temp_1.0'][t])
        data['temp12'][t] = texts(tmp['temp_1.2'][t])
        data['prompt_v1'][t] = texts(pv1[t])
        data['prompt_v2'][t] = texts(pv2[t])
        data['dds07'][t] = pool_trials(dyn['dds_alpha_0.5'], t)
        data['dds12'][t] = pool_trials(dds12['dds_alpha_0.5'], t)
        data['debate'][t] = pool_trials(dyn['debate'], t)
        data['map'][t] = pool_trials(dyn['map_elites'], t)
    # ---- new tasks
    keymap = {'ref07': ('independent', 'indep_t07'), 'temp10': ('independent', 'indep_t10'),
              'temp12': ('independent', 'indep_t12'), 'prompt_v1': ('independent', 'prompt_v1'),
              'prompt_v2': ('independent', 'prompt_v2'), 'dds07': ('loops', 'dds_a05_t07'),
              'dds12': ('loops', 'dds_a05_t12'), 'debate': ('loops', 'debate_t07'),
              'map': ('loops', 'map_elites_t07')}
    for t in NEW_TASKS:
        d = json.load(open(sorted(glob.glob(
            f'results/task_expansion/pilot_{t}_2*.json'))[-1]))
        for c, (sec, key) in keymap.items():
            if sec == 'independent':
                data[c][t] = texts(d[sec][key])
            else:
                data[c][t] = [r['text'] for tr in d[sec][key]
                              for rt in tr['response_texts'] for r in rt]
    return data


def participation_ratio(E):
    if len(E) < 3:
        return float('nan')
    ev = PCA(n_components=min(len(E), E.shape[1])).fit(E).explained_variance_
    return float((ev.sum() ** 2) / np.square(ev).sum())


def boot_se(E, rng):
    n = len(E)
    pr = np.array([participation_ratio(E[rng.randint(0, n, n)]) for _ in range(B)])
    return float(pr.std(ddof=1))


def escape_block(ref, test, rng, n_splits=ESC_SPLITS):
    """Paper's escape: cosine, split-half reference, eps = 2x median NN."""
    n = len(ref) // 2
    esc_t, esc_h = [], []
    for _ in range(n_splits):
        idx = rng.permutation(len(ref))
        R, H = ref[idx[:n]], ref[idx[n:2 * n]]
        rd = cosine_distances(R)
        np.fill_diagonal(rd, np.inf)
        eps = float(np.median(rd.min(axis=1))) * 2.0
        Tm = test[rng.choice(len(test), min(n, len(test)), replace=False)]
        esc_t.append(float(np.mean(cosine_distances(Tm, R).min(axis=1) > eps)))
        esc_h.append(float(np.mean(cosine_distances(H, R).min(axis=1) > eps)))
    return float(np.mean(esc_t)), float(np.mean(esc_h))


def captured(X, Vk):
    Xc = X - X.mean(0)
    tot = float((Xc ** 2).sum())
    inside = float(((Xc @ Vk) ** 2).sum())
    return inside / tot


def leakage_block(ref, sets, rng, n_splits=LEAK_SPLITS, k=LEAK_K):
    n = len(ref) // 2
    acc = {c: [] for c in sets}
    acc['held'] = []
    for _ in range(n_splits):
        idx = rng.permutation(len(ref))
        fit, held = ref[idx[:n]], ref[idx[n:2 * n]]
        Vk = PCA(n_components=k).fit(fit).components_.T
        acc['held'].append(captured(held, Vk))
        for c, X in sets.items():
            sub = X[rng.choice(len(X), min(n, len(X)), replace=False)]
            acc[c].append(captured(sub, Vk))
    held = float(np.mean(acc['held']))
    return {c: held - float(np.mean(acc[c])) for c in sets}


def paired(a, b, label):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_rel(a, b)
    try:
        _, wp = stats.wilcoxon(a, b)
    except ValueError:
        wp = float('nan')
    d = (a - b)
    return {'label': label, 'n': len(a), 't': round(float(t), 2), 'df': len(a) - 1,
            'p': round(float(p), 5), 'wilcoxon_p': round(float(wp), 5),
            'd': round(float(d.mean() / d.std(ddof=1)), 2),
            'sign': f'{int((d > 0).sum())}/{len(a)}'}


def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    rng = np.random.RandomState(SEED)
    print('loading & embedding 12 tasks x 9 conditions ...', flush=True)
    data = load_all()
    E = {c: {t: np.asarray(model.encode(data[c][t], show_progress_bar=False), dtype=float)
             for t in TASKS} for c in CONDS}
    out = {'n': {c: {t: len(E[c][t]) for t in TASKS} for c in CONDS}}

    # ---------- d_eff table with bootstrap CIs ----------
    print(f'\n=== d_eff (participation ratio, point [95%% CI], native n, B={B}) ==='.replace('%%', '%'), flush=True)
    deff = {c: {} for c in CONDS}
    hdr = f"{'task':<12}" + ''.join(f'{c:>20}' for c in CONDS)
    print(hdr)
    for t in TASKS:
        cells = []
        for c in CONDS:
            pt = participation_ratio(E[c][t])
            se = boot_se(E[c][t], rng)
            deff[c][t] = {'point': round(pt, 2), 'se': round(se, 3),
                          'ci': [round(pt - 1.96 * se, 2), round(pt + 1.96 * se, 2)]}
            cells.append(f'{pt:5.1f} [{pt-1.96*se:4.1f},{pt+1.96*se:4.1f}]')
        print(f'{t:<12}' + ''.join(f'{x:>20}' for x in cells), flush=True)
    means = {c: float(np.mean([deff[c][t]['point'] for t in TASKS])) for c in CONDS}
    print(f"{'MEAN':<12}" + ''.join(f"{means[c]:>20.1f}" for c in CONDS))
    out['d_eff'] = deff
    out['d_eff_mean'] = {c: round(means[c], 2) for c in CONDS}

    # ---------- escape ----------
    print('\n=== escape (cosine split-half, eps=2x median NN, %d splits) ===' % ESC_SPLITS, flush=True)
    esc = {}
    print(f"{'task':<12}{'ctrl':>8}{'dds07':>8}{'temp12':>8}{'prompt':>8}")
    for t in TASKS:
        ref = E['ref07'][t]
        e_d, e_h = escape_block(ref, E['dds07'][t], rng)
        e_t, _ = escape_block(ref, E['temp12'][t], rng)
        e_p, _ = escape_block(ref, E['prompt_v1'][t], rng)
        esc[t] = {'ctrl': round(e_h, 4), 'dds07': round(e_d, 4),
                  'temp12': round(e_t, 4), 'prompt_v1': round(e_p, 4)}
        print(f"{t:<12}{e_h:>8.3f}{e_d:>8.3f}{e_t:>8.3f}{e_p:>8.3f}", flush=True)
    out['escape'] = esc
    for k in ('ctrl', 'dds07', 'temp12', 'prompt_v1'):
        print(f"  mean {k}: {np.mean([esc[t][k] for t in TASKS]):.3f}")

    # ---------- leakage ----------
    print('\n=== leakage vs held-out control (K=%d, %d splits) ===' % (LEAK_K, LEAK_SPLITS), flush=True)
    leak = {}
    print(f"{'task':<12}{'dds07':>8}{'temp12':>8}{'pr_v1':>8}{'pr_v2':>8}")
    for t in TASKS:
        lb = leakage_block(E['ref07'][t],
                           {'dds07': E['dds07'][t], 'temp12': E['temp12'][t],
                            'prompt_v1': E['prompt_v1'][t], 'prompt_v2': E['prompt_v2'][t]},
                           rng)
        leak[t] = {k: round(v, 4) for k, v in lb.items()}
        print(f"{t:<12}{lb['dds07']:>8.3f}{lb['temp12']:>8.3f}"
              f"{lb['prompt_v1']:>8.3f}{lb['prompt_v2']:>8.3f}", flush=True)
    out['leakage'] = leak
    for k in ('dds07', 'temp12', 'prompt_v1', 'prompt_v2'):
        print(f"  mean {k}: {np.mean([leak[t][k] for t in TASKS]):.3f}")

    # ---------- task-level inference (n=12) ----------
    print('\n=== task-level inference (n=12) ===', flush=True)
    pt = lambda c: [deff[c][t]['point'] for t in TASKS]
    tests = [
        paired(pt('temp12'), pt('ref07'), 'd_eff: temp1.2 vs ref'),
        paired(pt('temp10'), pt('ref07'), 'd_eff: temp1.0 vs ref'),
        paired(pt('dds07'), pt('ref07'), 'd_eff: DDS@0.7 vs ref'),
        paired(pt('prompt_v1'), pt('ref07'), 'd_eff: prompt_v1 vs ref'),
        paired(pt('debate'), pt('ref07'), 'd_eff: debate vs ref'),
        paired(pt('map'), pt('ref07'), 'd_eff: map-elites vs ref'),
        paired(pt('dds12'), pt('temp12'), 'd_eff: DDS@1.2 vs temp1.2 (M4)'),
        paired(pt('dds12'), pt('ref07'), 'd_eff: DDS@1.2 vs ref (M4)'),
        paired([esc[t]['dds07'] for t in TASKS], [esc[t]['ctrl'] for t in TASKS],
               'escape: DDS vs ctrl'),
        paired([leak[t]['prompt_v1'] for t in TASKS], [leak[t]['temp12'] for t in TASKS],
               'leakage: prompt_v1 vs temp'),
        paired([leak[t]['temp12'] for t in TASKS], [leak[t]['dds07'] for t in TASKS],
               'leakage: temp vs DDS'),
        paired([leak[t]['prompt_v1'] for t in TASKS], [leak[t]['dds07'] for t in TASKS],
               'leakage: prompt_v1 vs DDS'),
    ]
    for r in tests:
        print(f"  {r['label']:<32} t({r['df']})={r['t']:>6} p={r['p']:<8} "
              f"wilcoxon p={r['wilcoxon_p']:<8} d={r['d']:>5} sign {r['sign']}")
    out['tests'] = tests

    # ---------- openness gradient ----------
    print('\n=== openness gradient (Spearman, pre-registered ranks) ===', flush=True)
    grad = {}
    for name, ranks in (('pre-registered', OPENNESS), ('alt: creative_2->ideation', OPENNESS_ALT)):
        xs = [ranks[t] for t in TASKS]
        for measure, ys in (('escape_dds_excess', [esc[t]['dds07'] - esc[t]['ctrl'] for t in TASKS]),
                            ('d_eff_ref', pt('ref07')),
                            ('temp_expansion_ratio', [deff['temp12'][t]['point'] / deff['ref07'][t]['point'] for t in TASKS])):
            rho, p = stats.spearmanr(xs, ys)
            grad[f'{name}|{measure}'] = {'rho': round(float(rho), 3), 'p': round(float(p), 4)}
            print(f"  [{name}] {measure:<24} rho={rho:+.3f} p={p:.4f}")
    out['gradient'] = grad

    json.dump(out, open('results/task_expansion/full12_analysis.json', 'w'), indent=1)
    print('\nsaved -> results/task_expansion/full12_analysis.json', flush=True)


if __name__ == '__main__':
    main()

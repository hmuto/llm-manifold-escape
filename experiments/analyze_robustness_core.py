#!/usr/bin/env python3
"""Robustness recomputations (all from existing response data).

 (1) Eigenvalue spectra of ref07 / temp12 / prompt_v1 / dds07 (matched n),
     per task + task-mean -> figures/fig_spectra.pdf
 (2) Centroid-shift decomposition of the radius measure:
     radius ratio, centroid-shift ratio, own-spread ratio
 (3) Cross-model leakage positive control: Claude Haiku 4.5
     (independent / DDS) measured against the GPT-4o-mini
     T=0.7 reference, same 40-split k=20 estimator
 (4) Bandwidth (h) sensitivity of density ranking on N=8
     subsets: Spearman rho vs h=0.3, top-1 agreement
 (5) 12-task version of the temperature figure
     -> figures/fig_temperature_expansion_12task.pdf

Estimators are copied verbatim from analyze_12task_full.py (paper's exact
procedures; leakage: 40 splits, k=20, held-out control; seed 0).
Output: results/robustness/robustness_core.json
"""
import os, sys, json, glob
import numpy as np
from scipy import stats

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": ["Arial", "Helvetica Neue", "Helvetica"],
                     "pdf.fonttype": 42,
                     "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold",
                     "mathtext.cal": "Arial",
                     "mathtext.sf": "Arial"})

LEAK_SPLITS, LEAK_K, SEED = 40, 20, 0
OLD_TASKS = ['creative_1', 'creative_2', 'problem_1', 'debate_1']
NEW_TASKS = ['reasoning_2', 'factual_1', 'factual_2', 'debate_2',
             'ideation_1', 'ideation_2', 'ideation_3', 'creative_3']
TASKS = OLD_TASKS + NEW_TASKS
CONDS = ['ref07', 'temp10', 'temp12', 'prompt_v1', 'dds07']
OUT = 'results/robustness'
os.makedirs(OUT, exist_ok=True)
os.makedirs('figures', exist_ok=True)


def texts(v):
    return [r['text'] if isinstance(r, dict) else r for r in v]


def pool_trials(cond_block, tid):
    for td in cond_block:
        if td['task_id'] == tid:
            return [r['text'] for tr in td['trials']
                    for rt in tr['response_texts'] for r in rt]
    return []


def load_gpt():
    data = {c: {} for c in CONDS}
    ref = json.load(open(sorted(glob.glob(
        'results/independent_scaling/independent_scaling_*.json'))[-1]))['responses_by_task']
    tmp = json.load(open(sorted(glob.glob(
        'results/temperature_expansion/temperature_expansion_2*.json'))[-1]))['responses_by_temp_task']
    pv1 = json.load(open(sorted(glob.glob(
        'results/prompt_expansion/prompt_expansion_2*.json'))[-1]))['responses_by_task']
    dyn = json.load(open(
        'results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json'))['conditions']
    for t in OLD_TASKS:
        data['ref07'][t] = texts(ref[t])
        data['temp10'][t] = texts(tmp['temp_1.0'][t])
        data['temp12'][t] = texts(tmp['temp_1.2'][t])
        data['prompt_v1'][t] = texts(pv1[t])
        data['dds07'][t] = pool_trials(dyn['dds_alpha_0.5'], t)
    keymap = {'ref07': ('independent', 'indep_t07'), 'temp10': ('independent', 'indep_t10'),
              'temp12': ('independent', 'indep_t12'), 'prompt_v1': ('independent', 'prompt_v1'),
              'dds07': ('loops', 'dds_a05_t07')}
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


def load_claude():
    d = json.load(open(sorted(glob.glob(
        'results/robustness_claude/robustness_claude_2*.json'))[-1]))['conditions']
    out = {}
    for name, key in [('claude_ind', 'independent'), ('claude_dds', 'dds_alpha_0.5')]:
        out[name] = {t: pool_trials(d[key], t) for t in OLD_TASKS}
    return out


def captured(X, Vk):
    Xc = X - X.mean(0)
    return float(((Xc @ Vk) ** 2).sum()) / float((Xc ** 2).sum())


def leakage_block(ref, sets, rng, n_splits=LEAK_SPLITS, k=LEAK_K):
    n = len(ref) // 2
    acc = {c: [] for c in sets}; acc['held'] = []
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


def participation_ratio(E):
    ev = PCA(n_components=min(len(E), E.shape[1])).fit(E).explained_variance_
    return float((ev.sum() ** 2) / np.square(ev).sum())


def cos_to_centroid(X, mu):
    # paper's radius: mean EUCLIDEAN distance to the centroid (analyze_support_vs_loop.py)
    return np.linalg.norm(X - mu, axis=1)


def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    rng = np.random.RandomState(SEED)
    print('loading GPT texts ...', flush=True)
    data = load_gpt()
    claude = load_claude()

    cache = os.path.join(OUT, 'emb_minilm.npz')
    E = {}
    if os.path.exists(cache):
        z = np.load(cache)
        E = {k: z[k] for k in z.files}
        print('embeddings loaded from cache', flush=True)
    else:
        for c in CONDS:
            for t in TASKS:
                E[f'{c}|{t}'] = model.encode(data[c][t], show_progress_bar=False)
                print(f'embedded {c}|{t} n={len(data[c][t])}', flush=True)
        for c in claude:
            for t in OLD_TASKS:
                E[f'{c}|{t}'] = model.encode(claude[c][t], show_progress_bar=False)
                print(f'embedded {c}|{t} n={len(claude[c][t])}', flush=True)
        np.savez_compressed(cache, **E)
    res = {}

    # ---------- (1) spectra ----------
    print('(1) spectra', flush=True)
    NSUB, NRS, NCOMP = 120, 20, 60
    spectra = {c: [] for c in ['ref07', 'temp12', 'prompt_v1', 'dds07']}
    for t in TASKS:
        for c in spectra:
            X = E[f'{c}|{t}']
            acc = np.zeros(NCOMP)
            for _ in range(NRS):
                sub = X[rng.choice(len(X), min(NSUB, len(X)), replace=False)]
                evr = PCA(n_components=NCOMP).fit(sub).explained_variance_ratio_
                acc += evr
            spectra[c].append(acc / NRS)
    spec_mean = {c: np.mean(np.stack(v), axis=0) for c, v in spectra.items()}
    res['spectra_task_mean'] = {c: [round(float(x), 6) for x in v] for c, v in spec_mean.items()}
    fig, axes2 = plt.subplots(1, 2, figsize=(10.5, 3.7))
    styles = {'ref07': ('Independent reference ($T=0.7$)', '#9aa0a6', '--'),
              'dds07': ('DDS selection ($T=0.7$)', '#1a73e8', '-'),
              'temp12': ('Temperature $T=1.2$', '#d93025', '-'),
              'prompt_v1': ('Distinctiveness prompt', '#188038', '-')}
    x = np.arange(1, NCOMP + 1)
    ax = axes2[0]
    for c, (lab, col, ls) in styles.items():
        ax.plot(x, spec_mean[c], ls, color=col, lw=1.9, label=lab)
    ax.set_yscale('log'); ax.set_xlabel('Principal component (rank)')
    ax.set_ylabel('Fraction of variance (task mean)')
    ax.set_title('(a) Eigenvalue spectra', fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax = axes2[1]
    ax.axhline(1.0, color='#9aa0a6', lw=1.2, ls='--')
    for c in ('dds07', 'temp12', 'prompt_v1'):
        lab, col, _ = styles[c]
        ax.plot(x, spec_mean[c] / spec_mean['ref07'], '-', color=col, lw=1.9, label=lab)
    ax.set_xlabel('Principal component (rank)')
    ax.set_ylabel('Share relative to reference')
    ax.set_title('(b) Variance share ÷ reference share (same rank)',
                 fontsize=10, loc='left')
    ax.annotate('flatter spectrum:\nleading share down, lower-ranked share up',
                xy=(44, 1.27), fontsize=8.2, color='#d93025', ha='center')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig('figures/fig_spectra.pdf'); plt.close(fig)

    # ---------- (2) centroid decomposition ----------
    print('(2) centroid decomposition', flush=True)
    dec = {}
    for c in ['dds07', 'temp12', 'prompt_v1']:
        rows = []
        for t in TASKS:
            R, X = E[f'ref07|{t}'], E[f'{c}|{t}']
            muR, muX = R.mean(0), X.mean(0)
            rad_R = float(cos_to_centroid(R, muR).mean())
            rows.append({'task': t,
                         'radius_ratio': float(cos_to_centroid(X, muR).mean()) / rad_R,
                         'centroid_shift_ratio': float(np.linalg.norm(muX - muR)) / rad_R,
                         'own_spread_ratio': float(cos_to_centroid(X, muX).mean()) / rad_R})
        dec[c] = {'per_task': rows,
                  'mean': {k: round(float(np.mean([r[k] for r in rows])), 3)
                           for k in ('radius_ratio', 'centroid_shift_ratio', 'own_spread_ratio')}}
    res['centroid_decomposition'] = dec

    # ---------- (3) Claude cross-model leakage ----------
    print('(3) cross-model leakage', flush=True)
    xm = {'per_task': {}, 'mean': {}}
    keys = ['claude_ind', 'claude_dds', 'dds07', 'temp12', 'prompt_v1']
    acc = {k: [] for k in keys}
    for t in OLD_TASKS:
        sets = {k: E[f'{k}|{t}'] for k in keys}
        lk = leakage_block(E[f'ref07|{t}'], sets, np.random.RandomState(SEED))
        xm['per_task'][t] = {k: round(v, 4) for k, v in lk.items()}
        for k in keys: acc[k].append(lk[k])
    xm['mean'] = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
    res['cross_model_leakage'] = xm

    # ---------- (4) h sensitivity ----------
    print('(4) h sensitivity', flush=True)
    H = [0.15, 0.2, 0.3, 0.45, 0.6]
    NSETS, NAG = 200, 8
    hs = {f'h={h}': {'spearman': [], 'top1': []} for h in H}
    for t in TASKS:
        X = E[f'ref07|{t}']
        D = cosine_distances(X)
        for _ in range(NSETS):
            idx = rng.choice(len(X), NAG, replace=False)
            d = D[np.ix_(idx, idx)]
            rho = {h: (np.exp(-0.5 * (d / h) ** 2).sum(1) - 1.0) for h in H}
            base = rho[0.3]
            for h in H:
                r = stats.spearmanr(rho[h], base).statistic
                hs[f'h={h}']['spearman'].append(float(r))
                hs[f'h={h}']['top1'].append(float(np.argmin(rho[h]) == np.argmin(base)))
    res['h_sensitivity'] = {k: {'mean_spearman': round(float(np.mean(v['spearman'])), 3),
                                'top1_agreement': round(float(np.mean(v['top1'])), 3)}
                            for k, v in hs.items()}

    # ---------- (5) 12-task temperature figure ----------
    print('(5) 12-task figure', flush=True)
    conds_fig = [('ref07', 'Independent\n$T=0.7$', '#9aa0a6'),
                 ('temp10', 'Independent\n$T=1.0$', '#f29900'),
                 ('temp12', 'Independent\n$T=1.2$', '#d93025'),
                 ('dds07', 'DDS selection\n$T=0.7$', '#1a73e8')]
    deff = {c: [] for c, _, _ in conds_fig}
    radius = {c: [] for c, _, _ in conds_fig}
    for t in TASKS:
        muR = E[f'ref07|{t}'].mean(0)
        radR = float(cos_to_centroid(E[f'ref07|{t}'], muR).mean())
        for c, _, _ in conds_fig:
            X = E[f'{c}|{t}']
            deff[c].append(participation_ratio(X))
            radius[c].append(float(cos_to_centroid(X, muR).mean()) / radR)
    res['fig12_deff_taskmean'] = {c: round(float(np.mean(v)), 2) for c, v in deff.items()}
    res['fig12_radius_taskmean'] = {c: round(float(np.mean(v)), 3) for c, v in radius.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, dat, ylab, title in [
            (axes[0], deff, 'Effective dimension $d_{\\mathrm{eff}}$',
             '(a) Effective dimension (matched $n=120$)'),
            (axes[1], radius, 'Radius ratio vs. $T=0.7$ reference',
             '(b) Mean radius from the $T=0.7$ centroid')]:
        xs = np.arange(len(conds_fig))
        means = [np.mean(dat[c]) for c, _, _ in conds_fig]
        sems = [np.std(dat[c], ddof=1) / np.sqrt(len(dat[c])) for c, _, _ in conds_fig]
        ax.bar(xs, means, yerr=sems, capsize=4,
               color=[col for _, _, col in conds_fig], width=0.62)
        ax.set_xticks(xs); ax.set_xticklabels([lab for _, lab, _ in conds_fig], fontsize=8.6)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=10, loc='left')
        ax.spines[['top', 'right']].set_visible(False)
    axes[1].axhline(1.0, color='#9aa0a6', lw=0.9, ls=':')
    fig.tight_layout()
    fig.savefig('figures/fig_temperature_expansion_12task.pdf'); plt.close(fig)

    json.dump(res, open(os.path.join(OUT, 'robustness_core.json'), 'w'), indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

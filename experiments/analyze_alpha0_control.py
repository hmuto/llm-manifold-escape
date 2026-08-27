#!/usr/bin/env python3
"""Matched random-selection control (alpha=0) for the DDS loop.

The dynamics experiment ran the identical loop machinery at alpha in
{0.0, 0.5, 1.0}: same N=8 agents, three rounds, five trials per task,
n_survive=5 sampled with replacement, same context flow and generation
budget. With constant Q the softmax at alpha=0 is uniform, so selection is
uniform random over the eight candidates. Comparing alpha=0.0 with
alpha=0.5 therefore isolates the causal effect of the density term with
every other loop ingredient matched (context length, survivor count,
information flow, budget).

Reference construction: round-0 responses are generated without context, so
they are draws from the same base distribution regardless of alpha. Pooling
round 0 across the three alpha conditions gives an era-, runner-, and
style-matched reference of 3 x 5 x 8 = 120 responses per task. Later-round
responses (rounds 1-2, 80 per condition per task) are scored against this
reference with the paper's estimators: out-of-reference rate (escape_block),
radius ratio, size-matched participation-ratio effective dimension, and
control-adjusted subspace leakage (leakage_block).

Output: results/robustness/alpha0_control.json
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sentence_transformers import SentenceTransformer
from analyze_robustness_core import participation_ratio, OUT, SEED
from analyze_12task_full import escape_block, leakage_block, paired

DYN = 'results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json'
CONDS = ['dds_alpha_0.0', 'dds_alpha_0.5', 'dds_alpha_1.0']
AGENTS, ROUNDS = 8, 3
SUBS = 50


def split_rounds(rounds):
    """response_texts is nested: three rounds, each a list of eight
    {agent_id, text} dicts."""
    assert len(rounds) == ROUNDS and all(len(r) == AGENTS for r in rounds)
    r0 = [d['text'] for d in rounds[0]]
    later = [d['text'] for r in rounds[1:] for d in r]
    return r0, later


def main():
    d = json.load(open(DYN))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    rng = np.random.RandomState(SEED)

    tasks = [e['task_id'] for e in d['conditions'][CONDS[0]]]
    res = {'config': {'source': DYN, 'conditions': CONDS, 'tasks': tasks,
                      'reference': 'pooled round-0 across alpha conditions '
                                   '(context-free draws), n=120 per task',
                      'test_sets': 'rounds 1-2 per condition, n=80'},
           'per_task': {}, 'tests': []}
    acc = {}
    for ti, task in enumerate(tasks):
        r0_texts, later = [], {c: [] for c in CONDS}
        for cond in CONDS:
            entry = d['conditions'][cond][ti]
            assert entry['task_id'] == task
            for tr in entry['trials']:
                r0, rest = split_rounds(tr['response_texts'])
                r0_texts += r0
                later[cond] += rest
        ref = model.encode(r0_texts, show_progress_bar=False)
        mu = ref.mean(0)
        ref_rad = float(np.mean(np.linalg.norm(ref - mu, axis=1)))
        embs = {c: model.encode(later[c], show_progress_bar=False)
                for c in CONDS}
        row = {'n_ref': len(ref)}
        for cond in CONDS:
            E = embs[cond]
            esc, held = escape_block(ref, E, rng)
            n = len(E)
            subs = [participation_ratio(ref[rng.choice(len(ref),
                                                       min(n, len(ref)),
                                                       replace=False)])
                    for _ in range(SUBS)]
            row[cond] = {
                'n': n,
                'oor': round(esc, 4), 'oor_held': round(held, 4),
                'radius_ratio': round(float(np.mean(
                    np.linalg.norm(E - mu, axis=1))) / ref_rad, 3),
                'deff': round(participation_ratio(E), 2),
                'deff_ref_at_n': round(float(np.mean(subs)), 2),
            }
            print(f'[{task}] {cond}: oor={esc:.3f} (held {held:.3f}) '
                  f"deff={row[cond]['deff']} vs ref@n "
                  f"{row[cond]['deff_ref_at_n']} rad={row[cond]['radius_ratio']}",
                  flush=True)
        leaks = leakage_block(ref, embs, rng)
        for cond in CONDS:
            row[cond]['leakage'] = round(leaks[cond], 4)
        res['per_task'][task] = row
        for cond in CONDS:
            for m in ('oor', 'radius_ratio', 'deff', 'leakage'):
                acc.setdefault((cond, m), []).append(row[cond][m])

    # scalar pairwise diversity recorded at run time (round_diversities)
    res['scalar_diversity'] = {}
    for cond in CONDS:
        per_task = []
        for e in d['conditions'][cond]:
            per_task.append(float(np.mean(
                [np.mean(tr['round_diversities']) for tr in e['trials']])))
        res['scalar_diversity'][cond] = [round(v, 4) for v in per_task]
        for ti, task in enumerate(tasks):
            acc.setdefault((cond, 'scalar_diversity'), []).append(per_task[ti])

    res['task_means'] = {f'{c}:{m}': round(float(np.mean(v)), 4)
                         for (c, m), v in acc.items()}
    from scipy import stats as _st
    tcrit = float(_st.t.ppf(0.95, len(tasks) - 1))
    for m in ('oor', 'radius_ratio', 'deff', 'leakage', 'scalar_diversity'):
        t = paired(acc[('dds_alpha_0.5', m)], acc[('dds_alpha_0.0', m)],
                   f'{m}: alpha 0.5 vs alpha 0 (n=4 tasks)')
        diffs = (np.array(acc[('dds_alpha_0.5', m)], float)
                 - np.array(acc[('dds_alpha_0.0', m)], float))
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
        t['upper95_one_sided'] = round(float(diffs.mean()) + tcrit * se, 4)
        res['tests'].append(t)
    out = os.path.join(OUT, 'alpha0_control.json')
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res['task_means'], indent=1))
    print(json.dumps(res['tests'], indent=1))
    print('Saved:', out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Full TOST reporting for the paper's quality-equivalence claims.

Assembles task-level mean G-Eval quality (mean over the 30 judged responses
per cell) for the independent T=0.7 reference, the T=1.2 condition, and the
distinctiveness prompt, on all twelve tasks: original four from
temperature_quality / prompt_quality, new eight from quality_newtasks.

For each contrast (T=1.2 - reference, prompt - reference) it reports the
task-level paired mean difference, its 90% confidence interval (the interval
matched to a 5%-level TOST), both one-sided p-values at the pre-specified
margin of +/-0.25 judge-scale points, the TOST p (max of the two), and the
conventional two-sided p, with n at both levels stated.

Output: results/robustness/quality_tost.json
"""

import os, sys, json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_robustness_core import OUT

MARGIN = 0.25
ORIG4 = ['creative_1', 'creative_2', 'problem_1', 'debate_1']
R = 'results'


def newest(path_glob):
    import glob
    return sorted(glob.glob(path_glob))[-1]


def main():
    tq = json.load(open(newest(f'{R}/temperature_expansion/temperature_quality_*.json')))
    pq = json.load(open(newest(f'{R}/prompt_expansion/prompt_quality_*.json')))
    nq = json.load(open(newest(f'{R}/task_expansion/quality_newtasks_*.json')))

    ref, temp, prompt = {}, {}, {}
    for t in ORIG4:
        ref[t] = float(np.mean(tq['scores']['temp_0.7'][t]))
        temp[t] = float(np.mean(tq['scores']['temp_1.2'][t]))
        prompt[t] = float(np.mean(pq['scores'][t]))
    for key, v in nq['scores'].items():
        cond, task = key.split('|')
        if cond == 'indep_t07':
            ref[task] = float(np.mean(v))
        elif cond == 'indep_t12':
            temp[task] = float(np.mean(v))
        elif cond == 'prompt_v1':
            prompt[task] = float(np.mean(v))

    tasks = sorted(ref)
    assert len(tasks) == 12 and set(temp) >= set(tasks) and set(prompt) >= set(tasks)

    res = {'config': {'margin': MARGIN, 'n_per_cell': 30,
                      'note': 'margin chosen before this analysis; '
                              'not preregistered',
                      'unit': 'task-level means, n=12 tasks'},
           'per_task': {t: {'ref': round(ref[t], 3), 'temp12': round(temp[t], 3),
                            'prompt': round(prompt[t], 3)} for t in tasks},
           'contrasts': []}
    for name, cond in (('temp12_minus_ref', temp), ('prompt_minus_ref', prompt)):
        d = np.array([cond[t] - ref[t] for t in tasks])
        n = len(d)
        m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
        tcrit = stats.t.ppf(0.95, n - 1)
        ci90 = (round(m - tcrit * se, 3), round(m + tcrit * se, 3))
        p_low = float(1 - stats.t.cdf((m + MARGIN) / se, n - 1))
        p_up = float(stats.t.cdf((m - MARGIN) / se, n - 1))
        t2, p2 = stats.ttest_1samp(d, 0)
        res['contrasts'].append({
            'contrast': name, 'n_tasks': n,
            'mean_diff': round(m, 3), 'se': round(se, 3), 'ci90': ci90,
            'p_lower_gt_minus_margin': round(p_low, 5),
            'p_upper_lt_plus_margin': round(p_up, 5),
            'p_tost': round(max(p_low, p_up), 5),
            'p_two_sided': round(float(p2), 4),
            'equivalent_at_margin': bool(max(p_low, p_up) < 0.05),
        })
    out = os.path.join(OUT, 'quality_tost.json')
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res['contrasts'], indent=1))
    print('Saved:', out)


if __name__ == '__main__':
    main()

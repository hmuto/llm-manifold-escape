#!/usr/bin/env python3
"""Leakage (directional novelty) under a second embedding FAMILY:
OpenAI text-embedding-3-small (1536-d), all 12 tasks.        [Major 7]

Same estimator as the paper (analyze_12task_full.py): reference T=0.7
split fit/held, top-k=20 PCA subspace, control-adjusted, 40 splits, seed 0.
Conditions: dds07, temp12, prompt_v1 vs ref07.
Output: results/robustness/leakage_openai.json (+ .npz cache)
"""
import os, sys, json, glob, time
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_robustness_core import (load_gpt, leakage_block, TASKS, SEED, OUT)

MODEL = 'text-embedding-3-small'
CONDS = ['ref07', 'temp12', 'prompt_v1', 'dds07']


def embed_all(data):
    cache = os.path.join(OUT, 'emb_openai_small.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    client = OpenAI()
    E = {}
    for c in CONDS:
        for t in TASKS:
            txts = [x[:8000] for x in data[c][t]]
            vecs = []
            for i in range(0, len(txts), 128):
                r = client.embeddings.create(model=MODEL, input=txts[i:i + 128])
                vecs.extend([d.embedding for d in r.data])
                time.sleep(0.2)
            E[f'{c}|{t}'] = np.asarray(vecs, dtype=np.float32)
            print(f'embedded {c}|{t} n={len(vecs)}', flush=True)
    np.savez_compressed(cache, **E)
    return E


def main():
    data = load_gpt()
    E = embed_all(data)
    per_task, acc = {}, {c: [] for c in ['dds07', 'temp12', 'prompt_v1']}
    for t in TASKS:
        sets = {c: E[f'{c}|{t}'] for c in acc}
        lk = leakage_block(E[f'ref07|{t}'], sets, np.random.RandomState(SEED))
        per_task[t] = {c: round(v, 4) for c, v in lk.items()}
        for c in acc: acc[c].append(lk[c])
    res = {'model': MODEL,
           'per_task': per_task,
           'mean': {c: round(float(np.mean(v)), 4) for c, v in acc.items()}}

    def paired(a, b, label):
        a, b = np.asarray(a), np.asarray(b)
        t, p = stats.ttest_rel(a, b)
        return {'label': label, 't': round(float(t), 2), 'df': len(a) - 1,
                'p': round(float(p), 5),
                'sign': f'{int(((a - b) > 0).sum())}/{len(a)}'}
    res['tests'] = [paired(acc['prompt_v1'], acc['dds07'], 'prompt vs dds'),
                    paired(acc['prompt_v1'], acc['temp12'], 'prompt vs temp'),
                    paired(acc['temp12'], acc['dds07'], 'temp vs dds')]
    json.dump(res, open(os.path.join(OUT, 'leakage_openai.json'), 'w'), indent=1)
    print(json.dumps(res['mean'], indent=1))
    for t in res['tests']: print(t)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

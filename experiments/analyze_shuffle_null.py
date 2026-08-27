#!/usr/bin/env python3
"""
Stronger null for the effective-dimension claim.

Isotropic random vectors are a weak null: sentence embeddings are known to be
anisotropic, so low effective dimension relative to isotropic random is partly a
generic property of the embedding space, not specific to task responses. We add a
null that PRESERVES each embedding dimension's marginal (hence the anisotropy) but
DESTROYS the cross-dimension correlation that produces low-dimensional task
structure: an independent per-column shuffle of the real embeddings. If the real
effective dimension is well below this shuffle null, the low-dimensional structure
is task-specific correlation, not just marginal anisotropy.
"""

import json
import numpy as np
from sklearn.decomposition import PCA

DDS_FILE = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
N_SHUFFLE = 10


def part_ratio(E):
    p = PCA(n_components=min(len(E), E.shape[1])).fit(E)
    ev = p.explained_variance_
    return float((ev.sum() ** 2) / np.square(ev).sum())


def pooled_texts():
    d = json.load(open(DDS_FILE))
    per_task = {}
    for cond in d["conditions"].values():
        for td in cond:
            tid = td["task_id"]; per_task.setdefault(tid, [])
            for tr in td["trials"]:
                for rt in tr.get("response_texts", []):
                    for r in rt:
                        per_task[tid].append(r["text"])
    return per_task


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = pooled_texts()
    rng = np.random.RandomState(0)

    print(f"{'task':<12} {'n':>5} {'d_eff real':>11} {'d_eff col-shuffle':>18}")
    reals, shufs = [], []
    for tid, tx in texts.items():
        E = np.asarray(model.encode(tx, show_progress_bar=False), dtype=float)
        real = part_ratio(E)
        sh = []
        for _ in range(N_SHUFFLE):
            S = np.empty_like(E)
            for j in range(E.shape[1]):
                S[:, j] = E[rng.permutation(len(E)), j]
            sh.append(part_ratio(S))
        shm = float(np.mean(sh))
        reals.append(real); shufs.append(shm)
        print(f"{tid:<12} {len(E):>5} {real:>11.1f} {shm:>18.1f}")
    print(f"{'mean':<12} {'':>5} {np.mean(reals):>11.1f} {np.mean(shufs):>18.1f}")
    print(f"\nreal / shuffle ratio (mean): {np.mean(reals)/np.mean(shufs):.2f}")
    print("If <1, task responses are lower-dimensional than a marginal-preserving,")
    print("correlation-destroying null => low dim is task-specific, not just anisotropy.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Effective-dimensionality analysis of LLM response embeddings.

Question (Pivot 2): the nominal embedding dimension is 384 (all-MiniLM) or
1536 (OpenAI). In such high dimensions, pairwise distances concentrate and
density estimation should break down (curse of dimensionality). Yet
density-dependent selection works. Why?

Hypothesis: LLM responses to a given task lie on a low-dimensional manifold,
so the *effective* dimension is far below the nominal dimension. We test this
by re-embedding the 2,560 responses from the main dynamics experiment and
computing effective-dimension measures per task and pooled.

Measures:
  - PCA explained variance -> #PCs for 90% / 95% variance
  - Participation ratio  PR = (sum lambda)^2 / sum(lambda^2)
  - Distance concentration  = std(pairwise) / mean(pairwise)
    (in high-dim iid data this -> 0; larger means more structure)
"""

import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

INPUT = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"


def participation_ratio(eigvals):
    eigvals = np.asarray(eigvals, dtype=float)
    return (eigvals.sum() ** 2) / (np.square(eigvals).sum())


def pcs_for_variance(explained_ratio, target):
    csum = np.cumsum(explained_ratio)
    return int(np.searchsorted(csum, target) + 1)


def concentration(embs):
    d = euclidean_distances(embs)
    triu = d[np.triu_indices(len(embs), k=1)]
    return float(np.std(triu) / np.mean(triu)), float(np.mean(triu))


def analyze(embs, label):
    embs = np.asarray(embs)
    n, dim = embs.shape
    pca = PCA(n_components=min(n, dim))
    pca.fit(embs)
    ev = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    pr = participation_ratio(ev)
    pc90 = pcs_for_variance(evr, 0.90)
    pc95 = pcs_for_variance(evr, 0.95)
    conc, meandist = concentration(embs)
    print(f"\n[{label}]  n={n}, nominal dim={dim}")
    print(f"  PCs for 90% var : {pc90}")
    print(f"  PCs for 95% var : {pc95}")
    print(f"  Participation ratio (effective dim): {pr:.2f}")
    print(f"  Top-5 explained variance ratio: {[round(x,3) for x in evr[:5]]}")
    print(f"  Distance concentration std/mean: {conc:.3f} (mean dist {meandist:.3f})")
    return {"n": n, "nominal_dim": dim, "pc90": pc90, "pc95": pc95,
            "participation_ratio": pr, "concentration": conc,
            "top5_evr": [float(x) for x in evr[:5]]}


def main():
    print("Loading responses:", INPUT)
    with open(INPUT) as f:
        data = json.load(f)

    # collect texts per task
    texts_by_task = {}
    for cond, cond_data in data["conditions"].items():
        for td in cond_data:
            tid = td["task_id"]
            for trial in td["trials"]:
                for rt in trial.get("response_texts", []):
                    for resp in rt:
                        texts_by_task.setdefault(tid, []).append(resp["text"])

    all_texts = [t for ts in texts_by_task.values() for t in ts]
    print(f"Total responses: {len(all_texts)}; tasks: {list(texts_by_task.keys())}")

    print("\nEmbedding with all-MiniLM-L6-v2 (384-dim, local)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    results = {"embedding_model": "all-MiniLM-L6-v2", "nominal_dim": 384, "per_task": {}}

    # per task
    for tid, texts in texts_by_task.items():
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        results["per_task"][tid] = analyze(embs, f"task={tid}")

    # pooled across all tasks
    embs_all = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
    results["pooled"] = analyze(embs_all, "ALL TASKS POOLED")

    # random baseline: iid Gaussian in 384-dim, same n, for concentration comparison
    rng = np.random.RandomState(0)
    n_ref = min(640, len(all_texts))
    gauss = rng.randn(n_ref, 384)
    gauss = gauss / np.linalg.norm(gauss, axis=1, keepdims=True)
    results["random_baseline"] = analyze(gauss, "RANDOM iid (384-dim, normalized)")

    out = Path("results/effective_dimension")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "effective_dimension_minilm.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out/'effective_dimension_minilm.json'}")

    # Summary interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    prs = [results["per_task"][t]["participation_ratio"] for t in texts_by_task]
    pc90s = [results["per_task"][t]["pc90"] for t in texts_by_task]
    print(f"Per-task participation ratio (effective dim): "
          f"{np.mean(prs):.1f} +/- {np.std(prs):.1f}  (nominal 384)")
    print(f"Per-task PCs for 90% variance: {np.mean(pc90s):.1f} +/- {np.std(pc90s):.1f}")
    print(f"Random iid baseline participation ratio: "
          f"{results['random_baseline']['participation_ratio']:.1f}")
    print(f"Random iid concentration: {results['random_baseline']['concentration']:.3f} "
          f"vs task-response concentration ~"
          f"{np.mean([results['per_task'][t]['concentration'] for t in texts_by_task]):.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Metric robustness: does the DDS effect survive under diversity metrics from
other families, not just mean pairwise cosine distance?

We recompute per-trial diversity of the cumulative response pools from the
dynamics experiment under four additional metrics and repeat the headline
contrasts (DDS alpha=0.5 vs Debate, the strongest convergence baseline; and DDS
vs MAP-Elites). Metrics span three families:
  - Vendi Score (embedding-spectral: effective number of distinct responses),
  - distinct-1, distinct-2 (lexical n-gram diversity),
  - self-BLEU (lexical overlap; LOWER = more diverse, so we report 1 - self-BLEU
    as a diversity score so that higher = more diverse for every metric).

DDS vs Debate and DDS vs MAP-Elites use equal-size pools (24 responses each), so
the size-sensitive metrics are compared fairly. If the DDS>Debate advantage and
the DDS~MAP equivalence reproduce under all metrics, the characterization is not
an artifact of the cosine-distance summary.
"""

import json
from collections import Counter
import numpy as np
from scipy import stats

DDS_FILE = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"


def cumulative_texts(cond_data):
    """Per trial: list of all response texts pooled across rounds."""
    out = []
    for task_data in cond_data:
        for trial in task_data["trials"]:
            texts = [r["text"] for rt in trial.get("response_texts", []) for r in rt]
            if len(texts) >= 2:
                out.append(texts)
    return out


def vendi_score(embs):
    X = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    n = len(X)
    K = X @ X.T / n
    w = np.linalg.eigvalsh(K)
    w = w[w > 1e-12]
    return float(np.exp(-np.sum(w * np.log(w))))


def distinct_n(texts, n):
    grams, total = set(), 0
    for t in texts:
        toks = t.lower().split()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i:i + n])); total += 1
    return len(grams) / total if total else 0.0


def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def _bleu(hyp, refs, max_n=4):
    """Sentence BLEU of hyp against a list of refs, add-1 smoothed precisions."""
    ht = hyp.lower().split()
    if not ht:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        hc = _ngrams(ht, n)
        if not hc:
            precisions.append(0.0); continue
        # max over refs of clipped counts
        max_ref = Counter()
        for r in refs:
            rc = _ngrams(r.lower().split(), n)
            for g, c in rc.items():
                if c > max_ref[g]:
                    max_ref[g] = c
        clipped = sum(min(c, max_ref[g]) for g, c in hc.items())
        total = sum(hc.values())
        precisions.append((clipped + 1.0) / (total + 1.0))   # add-1 smoothing
    # brevity penalty vs closest ref length
    hl = len(ht)
    rl = min((len(r.split()) for r in refs), key=lambda l: (abs(l - hl), l))
    bp = 1.0 if hl > rl else np.exp(1 - rl / hl) if hl > 0 else 0.0
    return float(bp * np.exp(np.mean(np.log(precisions))))


def self_bleu(texts):
    """Mean BLEU of each response against the others (lower = more diverse)."""
    if len(texts) < 2:
        return 0.0
    vals = [_bleu(texts[i], texts[:i] + texts[i + 1:]) for i in range(len(texts))]
    return float(np.mean(vals))


def per_trial_metrics(trials_texts, model):
    rows = {"cosine": [], "vendi": [], "distinct1": [], "distinct2": [], "inv_selfbleu": []}
    for texts in trials_texts:
        emb = np.asarray(model.encode(texts, show_progress_bar=False), dtype=float)
        from sklearn.metrics.pairwise import cosine_distances
        dm = cosine_distances(emb); iu = np.triu_indices(len(texts), k=1)
        rows["cosine"].append(float(dm[iu].mean()))
        rows["vendi"].append(vendi_score(emb))
        rows["distinct1"].append(distinct_n(texts, 1))
        rows["distinct2"].append(distinct_n(texts, 2))
        rows["inv_selfbleu"].append(1.0 - self_bleu(texts))
    return {k: np.asarray(v) for k, v in rows.items()}


def compare(a, b, label):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    t, p = stats.ttest_rel(a, b)
    d = (a - b).mean() / (a - b).std(ddof=1) if (a - b).std(ddof=1) > 0 else 0.0
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"    {label:<22} {a.mean():.3f} vs {b.mean():.3f}  "
          f"t({n-1})={t:+.2f} p={p:.4f} d={d:+.2f} {sig}")
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "p": float(p), "d": float(d)}


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    d = json.load(open(DDS_FILE))["conditions"]

    print("Computing per-trial metrics (cumulative pools)...")
    M = {c: per_trial_metrics(cumulative_texts(d[c]), model)
         for c in ["dds_alpha_0.5", "map_elites", "debate", "independent"]}

    out = {}
    for metric in ["cosine", "vendi", "distinct1", "distinct2", "inv_selfbleu"]:
        print(f"\n=== {metric} ===")
        out[metric] = {
            "DDS_vs_Debate": compare(M["dds_alpha_0.5"][metric], M["debate"][metric], "DDS vs Debate"),
            "DDS_vs_MAP":    compare(M["dds_alpha_0.5"][metric], M["map_elites"][metric], "DDS vs MAP-Elites"),
        }
    json.dump(out, open("results/support_vs_loop/metric_robustness.json", "w"), indent=2)
    print("\nSaved: results/support_vs_loop/metric_robustness.json")
    print("Higher = more diverse for every metric (self-BLEU reported as 1 - self-BLEU).")


if __name__ == "__main__":
    main()

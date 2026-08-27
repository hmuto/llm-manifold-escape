#!/usr/bin/env python3
"""
Embedding Robustness Check

Re-computes diversity metrics on the existing dynamics_mapelites data using
OpenAI's text-embedding-3-small in addition to the original all-MiniLM-L6-v2,
to verify that the main findings are not artifacts of a single embedding model.

Output: JSON with per-trial diversity computed under both embeddings, plus
correlation/agreement statistics between the two.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from openai import OpenAI

INPUT = "results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


def cosine_diversity(embeddings):
    if len(embeddings) < 2:
        return 0.0
    embs = np.asarray(embeddings)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sim = embs @ embs.T
    n = len(embs)
    upper = sim[np.triu_indices(n, k=1)]
    return float(np.mean(1.0 - upper))


def embed_batch(client, texts):
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        out.extend([np.array(d.embedding) for d in resp.data])
    return out


def main():
    print(f"Loading: {INPUT}")
    with open(INPUT) as f:
        data = json.load(f)

    client = OpenAI()

    all_texts = []
    text_index = []
    for cond_name, cond_data in data["conditions"].items():
        for task_data in cond_data:
            for t_idx, trial in enumerate(task_data["trials"]):
                texts_per_round = trial.get("response_texts", [])
                for r_idx, round_texts in enumerate(texts_per_round):
                    for resp in round_texts:
                        text_index.append((cond_name, task_data["task_id"], t_idx, r_idx))
                        all_texts.append(resp["text"])

    print(f"Total responses to embed: {len(all_texts):,}")
    print(f"Estimated cost: ~${len(all_texts) * 200 / 1_000_000 * 0.02:.4f}")

    print(f"Embedding with {EMBEDDING_MODEL}...")
    embeddings = embed_batch(client, all_texts)
    print(f"Got {len(embeddings)} embeddings, dim={len(embeddings[0])}")

    # Reorganize embeddings back into trial structure
    new_data = {
        "embedding_model": EMBEDDING_MODEL,
        "source_file": INPUT,
        "timestamp": datetime.now().isoformat(),
        "conditions": {},
    }

    cursor = 0
    for cond_name, cond_data in data["conditions"].items():
        new_data["conditions"][cond_name] = []
        for task_data in cond_data:
            task_out = {"task_id": task_data["task_id"], "trials": []}
            for trial in task_data["trials"]:
                texts_per_round = trial.get("response_texts", [])
                round_diversities_new = []
                for round_texts in texts_per_round:
                    n_in_round = len(round_texts)
                    round_embs = embeddings[cursor:cursor + n_in_round]
                    cursor += n_in_round
                    round_diversities_new.append(cosine_diversity(round_embs))
                task_out["trials"].append({
                    "round_diversities_new": round_diversities_new,
                    "round_diversities_orig": trial["round_diversities"],
                    "final_diversity_new": round_diversities_new[-1],
                    "final_diversity_orig": trial["final_diversity"],
                })
            new_data["conditions"][cond_name].append(task_out)

    output = Path(f"results/embedding_robustness/embedding_robustness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(new_data, f, indent=2, default=str)
    print(f"Saved: {output}")

    # Summary
    print("\n" + "="*60)
    print("Per-condition: orig (MiniLM) vs new (text-embedding-3-small)")
    print("="*60)
    from scipy import stats as scipy_stats
    for cond_name, cond_data in new_data["conditions"].items():
        orig_finals = []
        new_finals = []
        for task_data in cond_data:
            for trial in task_data["trials"]:
                orig_finals.append(trial["final_diversity_orig"])
                new_finals.append(trial["final_diversity_new"])
        if not orig_finals:
            continue
        r, p = scipy_stats.pearsonr(orig_finals, new_finals)
        print(f"  {cond_name}: orig={np.mean(orig_finals):.4f}+/-{np.std(orig_finals,ddof=1):.4f} | "
              f"new={np.mean(new_finals):.4f}+/-{np.std(new_finals,ddof=1):.4f} | "
              f"Pearson r={r:.3f} (p={p:.4f})")

    # Pairwise comparison preserved across embeddings?
    print("\nKey claim check: DDS alpha=0.5 vs Debate (paired t-test)")
    for emb_label, key in [("orig (MiniLM)", "final_diversity_orig"),
                          ("new (3-small)", "final_diversity_new")]:
        dds = []
        debate = []
        for task_data in new_data["conditions"]["dds_alpha_0.5"]:
            for trial in task_data["trials"]:
                dds.append(trial[key])
        for task_data in new_data["conditions"]["debate"]:
            for trial in task_data["trials"]:
                debate.append(trial[key])
        n_pair = min(len(dds), len(debate))
        t, p = scipy_stats.ttest_rel(dds[:n_pair], debate[:n_pair])
        diff = np.array(dds[:n_pair]) - np.array(debate[:n_pair])
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {emb_label}: t={t:.3f}, p={p:.4f}, d={d:.3f} ({sig})")


if __name__ == "__main__":
    main()

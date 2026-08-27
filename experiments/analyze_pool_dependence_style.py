#!/usr/bin/env python3
"""Pool-dependence and style-drift analyses on the DDS pool (no API calls).

M1 (pool dependence):
  - exact-duplicate count in each DDS pool (expected 0: the loop regenerates,
    it never copies survivors into the pool)
  - near-duplicate rate at cosine >= 0.99 and >= 0.95 (vs the same rate
    inside the independent reference, as a base rate)
  - d_eff after greedy near-duplicate removal, against an independent
    reference subsampled to the same reduced n (50 draws)

M2 (style drift vs tail reach):
  - OOR of the DDS pool split by round (round 0 has no context and is an
    independent draw by construction; rounds 1-2 are loop-conditioned)
  - response-length statistics per condition and per round
  - length-matched OOR: DDS responses whose lengths fall inside the
    reference's interquartile range

12 tasks, all-MiniLM-L6-v2, embeddings from the robustness cache.
Output: results/robustness/pool_dependence_style.json
"""

import os, sys, json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_robustness_core import load_gpt, participation_ratio, TASKS, SEED, OUT
from analyze_12task_full import escape_block

TRIALS, ROUNDS, AGENTS = 5, 3, 8


def dedup_greedy(E, thr):
    keep = []
    for i in range(len(E)):
        if all(float(E[i] @ E[j]) / (np.linalg.norm(E[i]) * np.linalg.norm(E[j]) + 1e-12) < thr
               for j in keep):
            keep.append(i)
    return keep


def main():
    z = np.load(os.path.join(OUT, "emb_minilm.npz"))
    data = load_gpt()
    rng = np.random.RandomState(SEED)
    res = {"per_task": {}, }
    acc = {}

    for t in TASKS:
        ref = z[f"ref07|{t}"]
        dds = z[f"dds07|{t}"]
        dtxt = data["dds07"][t]
        rtxt = data["ref07"][t]
        assert len(dds) == TRIALS * ROUNDS * AGENTS == len(dtxt)
        row = {}

        # ---- M1: duplicates ----
        row["exact_dup"] = len(dtxt) - len(set(dtxt))
        S = cosine_similarity(dds); np.fill_diagonal(S, -1)
        Sr = cosine_similarity(ref); np.fill_diagonal(Sr, -1)
        for thr in (0.99, 0.95):
            row[f"neardup_dds_{thr}"] = round(float((S.max(1) >= thr).mean()), 4)
            row[f"neardup_ref_{thr}"] = round(float((Sr.max(1) >= thr).mean()), 4)
        # dedup d_eff at 0.95 (aggressive) with size-matched reference
        keep = dedup_greedy(dds, 0.95)
        n = len(keep)
        row["dedup95_n"] = n
        row["deff_dds_dedup95"] = round(participation_ratio(dds[keep]), 2)
        subs = [participation_ratio(ref[rng.choice(len(ref), n, replace=False)])
                for _ in range(50)]
        row["deff_ref_at_n"] = round(float(np.mean(subs)), 2)

        # ---- M2: rounds ----
        rounds = np.array([(i % (ROUNDS * AGENTS)) // AGENTS for i in range(len(dds))])
        for r in range(ROUNDS):
            esc, held = escape_block(ref, dds[rounds == r], rng)
            row[f"oor_round{r}"] = round(esc, 4)
        row["oor_held"] = round(held, 4)

        # ---- M2: lengths ----
        L_ref = np.array([len(x) for x in rtxt]); L_dds = np.array([len(x) for x in dtxt])
        row["len_ref_mean"] = int(L_ref.mean()); row["len_dds_mean"] = int(L_dds.mean())
        for r in range(ROUNDS):
            row[f"len_dds_round{r}"] = int(L_dds[rounds == r].mean())
        q1, q3 = np.percentile(L_ref, [25, 75])
        inside = (L_dds >= q1) & (L_dds <= q3)
        row["len_matched_n"] = int(inside.sum())
        if inside.sum() >= 10:
            esc_m, _ = escape_block(ref, dds[inside], rng)
            row["oor_len_matched"] = round(esc_m, 4)

        res["per_task"][t] = row
        for k, v in row.items():
            if isinstance(v, (int, float)):
                acc.setdefault(k, []).append(v)

    res["task_means"] = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
    out = os.path.join(OUT, "pool_dependence_style.json")
    json.dump(res, open(out, "w"), indent=1)
    print("=== task means (12 tasks) ===")
    for k in sorted(res["task_means"]):
        print(f"  {k:22s} {res['task_means'][k]}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Model-drift check (task_expansion_plan.md §6/§7-4).

Regenerates independent N=128 @ T=0.7 for the ORIGINAL four tasks and compares
against the archived reference (results/independent_scaling/*.json, 2026-07-02):
participation-ratio d_eff, mean pairwise cosine distance, cross escape
(new-vs-old reference and old-vs-new), and radius ratio. If these match, the
new-task runs (July) can be pooled with the archived reference in one analysis.
"""
import sys
import os
import json
import glob
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.llm_agent import MultiAgentSystem, AgentConfig, EXAMPLE_TASKS

MODEL = "gpt-4o-mini"
MAX_TOKENS = 512
N_INDEP = 128
RETRIES = 6


def gen_retry(agent, prompt):
    delay = 2.0
    for a in range(RETRIES):
        try:
            return agent.generate(prompt)
        except Exception as e:
            if a == RETRIES - 1:
                raise
            print(f"    retry {a+1}: {type(e).__name__}; wait {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30)


def main():
    out_dir = Path("results/task_expansion")
    out_dir.mkdir(parents=True, exist_ok=True)
    of = out_dir / f"drift_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Resume support
    old_files = sorted(glob.glob(str(out_dir / "drift_check_2*.json")))
    if old_files:
        res = json.load(open(old_files[-1]))
        print(f"Resuming from {old_files[-1]}", flush=True)
    else:
        res = {"config": {"model": MODEL, "max_tokens": MAX_TOKENS,
                          "n_indep": N_INDEP, "temperature": 0.7},
               "timestamp": datetime.now().isoformat(),
               "responses_by_task": {}}

    tasks = EXAMPLE_TASKS[:4]
    cfg = AgentConfig(agent_id=0, backend="openai", model=MODEL,
                      temperature=0.7, max_tokens=MAX_TOKENS)
    system = None
    for task in tasks:
        got = res["responses_by_task"].get(task.task_id, [])
        if len(got) >= N_INDEP:
            print(f"[{task.task_id}] complete, skip", flush=True)
            continue
        if system is None:
            system = MultiAgentSystem(n_agents=N_INDEP, agent_config_template=cfg,
                                      embedding_model="all-MiniLM-L6-v2")
        print(f"[{task.task_id}] generating {N_INDEP - len(got)}...", flush=True)
        texts = list(got)
        for i in range(len(got), N_INDEP):
            texts.append({"agent_id": i, "text": gen_retry(system.agents[i], task.prompt)})
            if (i + 1) % 32 == 0:
                print(f"    {i+1}/{N_INDEP}", flush=True)
                res["responses_by_task"][task.task_id] = texts
                json.dump(res, open(of, "w"), indent=1)
        res["responses_by_task"][task.task_id] = texts
        json.dump(res, open(of, "w"), indent=1)
    json.dump(res, open(of, "w"), indent=1)

    # ---------------- comparison vs archive ----------------
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    model = SentenceTransformer('all-MiniLM-L6-v2')
    rng = np.random.default_rng(0)

    def embed(texts):
        return np.asarray(model.encode(texts, show_progress_bar=False))

    def d_eff(X):
        ev = PCA(n_components=min(len(X) - 1, X.shape[1])).fit(X).explained_variance_
        return float(ev.sum() ** 2 / (ev ** 2).sum())

    def pairdiv(X):
        from sklearn.metrics.pairwise import cosine_distances
        dm = cosine_distances(X)
        return float(dm[np.triu_indices(len(X), 1)].mean())

    def cross_escape(ref, test, n_splits=50):
        n = len(ref) // 2
        outs = []
        for _ in range(n_splits):
            idx = rng.permutation(len(ref))
            R = ref[idx[:n]]
            d_RR = np.linalg.norm(R[:, None] - R[None], axis=-1)
            np.fill_diagonal(d_RR, np.inf)
            eps = 2 * np.median(d_RR.min(1))
            Tm = test[rng.choice(len(test), n, replace=False)]
            outs.append((np.linalg.norm(Tm[:, None] - R[None], axis=-1).min(1) > eps).mean())
        return float(np.mean(outs))

    arch_files = sorted(glob.glob("results/independent_scaling/independent_scaling_*.json"))
    arch = json.load(open(arch_files[-1]))["responses_by_task"]

    def texts_of(v):
        return [r["text"] if isinstance(r, dict) else r for r in v]

    cmp = {}
    print("\n===== DRIFT CHECK (new July vs archived reference) =====")
    for task in tasks:
        tid = task.task_id
        new = embed(texts_of(res["responses_by_task"][tid]))
        old = embed(texts_of(arch[tid]))
        c_new, c_old = new.mean(0), old.mean(0)
        r = {
            "d_eff_old": round(d_eff(old), 1),
            "d_eff_new": round(d_eff(new), 1),
            "pairdiv_old": round(pairdiv(old), 4),
            "pairdiv_new": round(pairdiv(new), 4),
            "escape_new_vs_old": round(cross_escape(old, new), 3),
            "escape_old_vs_old": round(cross_escape(old, old), 3),
            "centroid_shift": round(float(np.linalg.norm(c_new - c_old)), 4),
        }
        cmp[tid] = r
        print(f"[{tid}] d_eff {r['d_eff_old']} -> {r['d_eff_new']} | "
              f"pairdiv {r['pairdiv_old']:.3f} -> {r['pairdiv_new']:.3f} | "
              f"esc(new|old)={r['escape_new_vs_old']:.3f} vs esc(old|old)={r['escape_old_vs_old']:.3f} | "
              f"centroid shift {r['centroid_shift']:.3f}", flush=True)

    res["comparison"] = cmp
    json.dump(res, open(of, "w"), indent=1)
    print(f"\nSaved: {of}\nDone: {datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()

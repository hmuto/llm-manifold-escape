#!/usr/bin/env python3
"""
M5: does the DIMENSIONAL claim (not just the diversity contrast) replicate on Claude?

Contribution 5 says the picture replicates on a second backend, but R1 only
replicated the diversity contrast; d_eff was measured on GPT-4o-mini alone. The
Claude Haiku 4.5 robustness run saved response texts for DDS (alpha=0.5),
Debate, and Independent, so we can compute the participation-ratio effective
dimension on Claude and check the confinement claim: DDS should NOT clear the
Independent reference. (Claude has no temperature condition, so only the
selection-confinement pole is testable here, not the temperature-clears pole.)
Matched n per task; bootstrap 95% CI.
"""

import sys, os, json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLAUDE = "results/robustness_claude/robustness_claude_20260426_141528.json"
B = 2000
SEED = 0


def load(cond):
    d = json.load(open(CLAUDE)); pool = {}
    for td in d["conditions"][cond]:
        t = td["task_id"]; pool.setdefault(t, [])
        for tr in td["trials"]:
            for rnd in tr.get("response_texts", []):
                pool[t].extend(r["text"] for r in rnd)
    return pool


def pr(E):
    if len(E) < 3:
        return float("nan")
    ev = PCA(n_components=min(len(E), E.shape[1])).fit(E).explained_variance_
    return float((ev.sum() ** 2) / np.square(ev).sum())


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda ts: np.asarray(model.encode(ts, show_progress_bar=False), dtype=float)

    conds = {"independent": load("independent"), "DDS (sel.)": load("dds_alpha_0.5"),
             "debate": load("debate")}
    tasks = list(conds["independent"].keys())
    E = {c: {t: enc(conds[c][t]) for t in tasks} for c in conds}
    nmatch = {t: min(len(E[c][t]) for c in conds) for t in tasks}
    print(f"backend = Claude Haiku 4.5; embedding all-MiniLM-L6-v2; "
          f"matched n/task = { {t: nmatch[t] for t in tasks} }; B={B}\n")

    def mean_ci(c):
        r2 = np.random.RandomState(SEED + 1)
        pts = [pr(E[c][t][r2.choice(len(E[c][t]), nmatch[t], replace=False)]) for t in tasks]
        draws = np.array([np.mean([pr(E[c][t][r2.randint(0, len(E[c][t]), nmatch[t])])
                                   for t in tasks]) for _ in range(B)])
        return float(np.mean(pts)), float(draws.std(ddof=1))

    ref_pt, ref_se = mean_ci("independent")
    ref_lo, ref_hi = ref_pt - 1.96 * ref_se, ref_pt + 1.96 * ref_se
    print(f"{'condition':<16} {'d_eff [95% CI]':>18} {'ratio':>7}  verdict")
    print(f"{'independent':<16} {f'{ref_pt:.1f} [{ref_lo:.1f},{ref_hi:.1f}]':>18} {'1.00x':>7}")
    out = {"backend": "claude-haiku-4-5", "reference": {"point": ref_pt, "se": ref_se}, "conds": {}}
    for c in ["DDS (sel.)", "debate"]:
        pt, s = mean_ci(c); lo, hi = pt - 1.96 * s, pt + 1.96 * s
        clears = lo > ref_hi
        v = "CLEARS ref (adds dims)" if clears else ("overlaps ref (flat)" if not (hi < ref_lo) else "below ref")
        out["conds"][c] = {"point": pt, "se": s, "ratio": pt / ref_pt, "clears_ref": clears}
        print(f"{c:<16} {f'{pt:.1f} [{lo:.1f},{hi:.1f}]':>18} {f'{pt/ref_pt:.2f}x':>7}  {v}")

    Path("results/robustness_claude/claude_effdim.json").write_text(json.dumps(out, indent=2))
    print("\nSaved: results/robustness_claude/claude_effdim.json")
    print("Read: if DDS does NOT clear the Independent reference on Claude, the")
    print("no-new-dimensions confinement replicates on a second backend (M5).")


if __name__ == "__main__":
    main()

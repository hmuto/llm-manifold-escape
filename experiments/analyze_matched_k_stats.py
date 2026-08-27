#!/usr/bin/env python3
"""
Inferential statistics for the decisive matched-k comparison.

The central claim ("selection does not exceed independent sampling") rests on the
matched-k coverage gap (DDS - independent). The gap needs inferential support,
not point estimates alone. For each k and pooled over k we report, over the four
tasks (the unit of analysis, n=4):
  - mean gap and 95% CI (t),
  - a ONE-SIDED test of H1: gap > 0 (does DDS exceed independent?),
  - a TOST equivalence test at a substantive margin (+/- 10 percentage points),
  - a sign test over all (task, k) cells (is the gap consistently negative?).
"""

import json
import numpy as np
from scipy import stats

d = json.load(open("results/support_vs_loop/matched_k_sweep.json"))
ks = d["ks"]; tasks = list(d["per_task"].keys())
MARGIN = 0.10   # equivalence margin: 10 percentage points of coverage


def gaps_at(k):
    return np.array([d["per_task"][t]["by_k"][str(k)]["diff"] for t in tasks])


def tost(x, margin):
    # two one-sided t-tests; equivalence p = max of the two one-sided p-values
    n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n); df = n - 1
    t_low = (m - (-margin)) / se        # H0: mean <= -margin
    p_low = stats.t.sf(t_low, df)
    t_up = (m - margin) / se            # H0: mean >= +margin
    p_up = stats.t.cdf(t_up, df)
    return max(p_low, p_up)


print(f"Matched-k gap (DDS - independent coverage), n={len(tasks)} tasks per k")
print(f"{'k':>4} {'mean gap':>9} {'95% CI':>18} {'p(exceed,1-sided)':>18} {'TOST p(<10pp)':>14}")
allg = []
for k in ks:
    g = gaps_at(k); allg.append(g)
    m = g.mean(); se = g.std(ddof=1) / np.sqrt(len(g)); df = len(g) - 1
    ci = stats.t.interval(0.95, df, loc=m, scale=se)
    p_exceed = stats.ttest_1samp(g, 0).pvalue / 2 if m > 0 else 1 - stats.ttest_1samp(g, 0).pvalue / 2
    p_tost = tost(g, MARGIN)
    print(f"{k:>4} {m*100:>+8.1f}pp [{ci[0]*100:>+6.1f},{ci[1]*100:>+6.1f}]pp "
          f"{p_exceed:>17.3f} {p_tost:>14.3f}")

allg = np.concatenate(allg)
neg = int((allg < 0).sum()); tot = len(allg)
sign_p = stats.binomtest(neg, tot, 0.5, alternative="greater").pvalue
print(f"\nSign test over all {tot} (task,k) cells: {neg}/{tot} gaps negative, "
      f"binomial p = {sign_p:.2e}")
print(f"Pooled mean gap: {allg.mean()*100:+.1f}pp "
      f"(one-sided p that DDS exceeds independent = "
      f"{(stats.ttest_1samp(allg,0).pvalue/2 if allg.mean()>0 else 1-stats.ttest_1samp(allg,0).pvalue/2):.3f})")
print("\nReading: DDS does not exceed independent (one-sided tests never reject),")
print("and the gap is small, consistently NEGATIVE (weak confinement), equivalent")
print("to independent within a 10-pp margin at the larger k.")

#!/usr/bin/env python3
"""Decorrelation and feedback controls: (a) per-task decorrelation with paired tests,
(b) Debate as a feedback-without-selection control for the OOR,
(c) task-level test for the unique-idea comparison.

Output: results/robustness/decorrelation_controls.json
"""
import os, sys, json, glob
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_robustness_core import participation_ratio, TASKS, OLD_TASKS, NEW_TASKS, SEED, OUT
from analyze_12task_full import escape_block

TR, RD, AG = 5, 3, 8


def load_debate():
    out = {}
    dyn = json.load(open('results/dynamics_mapelites/dynamics_mapelites_20260207_204808.json'))['conditions']
    for td in dyn['debate']:
        if td['task_id'] in OLD_TASKS:
            out[td['task_id']] = [r['text'] for tr in td['trials']
                                  for rt in tr['response_texts'] for r in rt]
    for t in NEW_TASKS:
        d = json.load(open(sorted(glob.glob(f'results/task_expansion/pilot_{t}_2*.json'))[-1]))
        out[t] = [r['text'] for tr in d['loops']['debate_t07']
                  for rt in tr['response_texts'] for r in rt]
    return out


def main():
    rng = np.random.RandomState(SEED)
    z = np.load(os.path.join(OUT, 'emb_minilm.npz'))
    res = {}

    # ---------- (a) per-task decorrelation, dynamics 4 tasks ----------
    dec = {"full": {}, "final_round": {}, "per_trial": {}}
    for t in OLD_TASKS:
        dds, ref = z[f'dds07|{t}'], z[f'ref07|{t}']
        rounds = np.array([(i % (RD * AG)) // AG for i in range(len(dds))])
        def ref_at(n, draws=50):
            return float(np.mean([participation_ratio(ref[rng.choice(len(ref), n, False)])
                                  for _ in range(draws)]))
        dec["full"][t] = (round(participation_ratio(dds), 2), round(ref_at(len(dds)), 2))
        fin = dds[rounds == RD - 1]
        dec["final_round"][t] = (round(participation_ratio(fin), 2), round(ref_at(len(fin)), 2))
        trial_vals = [participation_ratio(dds[i * RD * AG:(i + 1) * RD * AG]) for i in range(TR)]
        dec["per_trial"][t] = (round(float(np.mean(trial_vals)), 2), round(ref_at(RD * AG), 2))
    for variant, d in dec.items():
        a = np.array([v[0] for v in d.values()]); b = np.array([v[1] for v in d.values()])
        tt, p = stats.ttest_rel(a, b)
        d["paired"] = {"ratio": round(float((a / b).mean()), 3),
                       "t3": round(float(tt), 2), "p": round(float(p), 4),
                       "sign_above": f"{int((a > b).sum())}/4"}
    res["decorrelation"] = dec

    # ---------- (b) Debate OOR, 12 tasks ----------
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    deb_txt = load_debate()
    deb = {"per_task": {}}
    accs = {k: [] for k in ("full", "r0", "r1", "r2", "held")}
    for t in TASKS:
        E = np.asarray(model.encode(deb_txt[t], show_progress_bar=False), dtype=np.float32)
        ref = z[f'ref07|{t}']
        rounds = np.array([(i % (RD * AG)) // AG for i in range(len(E))])
        esc, held = escape_block(ref, E, rng)
        row = {"full": round(esc, 4), "held": round(held, 4)}
        for r in range(RD):
            e, _ = escape_block(ref, E[rounds == r], rng)
            row[f"r{r}"] = round(e, 4)
        deb["per_task"][t] = row
        for k in accs: accs[k].append(row[k if k != "held" else "held"])
        print(f"[debate] {t}: full={row['full']} r0={row['r0']} r1={row['r1']} r2={row['r2']}", flush=True)
    deb["means"] = {k: round(float(np.mean(v)), 4) for k, v in accs.items()}
    res["debate_oor"] = deb

    # ---------- (c) unique-ideas task-level test ----------
    ui = json.load(open(os.path.join(OUT, 'unique_ideas_temp.json')))
    a = np.array([np.mean(v["temp12"]) for v in ui["per_task"].values()])
    b = np.array([np.mean(v["ref07"]) for v in ui["per_task"].values()])
    tt, p = stats.ttest_rel(a, b)
    res["unique_ideas_task_level"] = {
        "task_means_ref": [round(float(x), 1) for x in b],
        "task_means_temp": [round(float(x), 1) for x in a],
        "t3": round(float(tt), 2), "p": round(float(p), 4),
        "sign": f"{int((a > b).sum())}/4"}

    out = os.path.join(OUT, 'decorrelation_controls.json')
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'debate_oor'}, indent=1))
    print("debate means:", res["debate_oor"]["means"])
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()

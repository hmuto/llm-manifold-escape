#!/usr/bin/env python3
"""(1) Embedding-input truncation audit: all-MiniLM-L6-v2 truncates at 256
word-piece tokens; report mean token counts and the fraction of responses
exceeding the limit, per condition (12 tasks).
(2) Leakage k-dependence: control-adjusted leakage at k in {10,20,30,40}.
Output: results/robustness/truncation_ksweep.json
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_robustness_core import load_gpt, leakage_block, TASKS, SEED, OUT

def main():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer('all-MiniLM-L6-v2')
    tok = st.tokenizer
    print("max_seq_length:", st.max_seq_length, flush=True)
    data = load_gpt()
    res = {"max_seq_length": int(st.max_seq_length), "truncation": {}, "ksweep": {}}
    for c in ("ref07", "dds07", "temp12", "prompt_v1"):
        toks, over = [], []
        for t in TASKS:
            n = [len(tok.encode(x, add_special_tokens=True)) for x in data[c][t]]
            toks += n
            over += [x > st.max_seq_length for x in n]
        res["truncation"][c] = {"mean_tokens": round(float(np.mean(toks)), 1),
                                "frac_truncated": round(float(np.mean(over)), 4)}
        print(c, res["truncation"][c], flush=True)

    z = np.load(os.path.join(OUT, "emb_minilm.npz"))
    for k in (10, 20, 30, 40):
        rng = np.random.RandomState(SEED)
        acc = {c: [] for c in ("dds07", "temp12", "prompt_v1")}
        for t in TASKS:
            lk = leakage_block(z[f"ref07|{t}"],
                               {c: z[f"{c}|{t}"] for c in acc}, rng, k=k)
            for c in acc: acc[c].append(lk[c])
        res["ksweep"][k] = {c: round(float(np.mean(v)), 3) for c, v in acc.items()}
        print("k=", k, res["ksweep"][k], flush=True)
    json.dump(res, open(os.path.join(OUT, "truncation_ksweep.json"), "w"), indent=1)
    print("Saved.")

if __name__ == "__main__":
    main()

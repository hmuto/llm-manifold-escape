# Can an LLM Escape Its Own Manifold?

Code and data for the paper **"Can an LLM Escape Its Own Manifold? Three
Geometric Forms of Diversity Expansion"**
(Muto, Ogi, and Yakoh; manuscript under review).

We decompose "escaping the manifold" of an LLM's output distribution into
three forms: **tail reach** (occupying low-density regions of the accessible
support), **dimensional expansion** (raising the effective dimension of the
occupied subspace), and **directional novelty** (adding variance along new
semantic directions). Comparing density-dependent selection (DDS, a fitness-
sharing rule applied to a multi-agent regeneration loop) against two external
interventions — higher decoding temperature and a distinctiveness prompt — we
find that selection reaches the tails but changes neither dimension count nor
direction, whereas temperature expands dimensions and the prompt relocates
outputs along novel directions. Experiments use GPT-4o-mini for generation,
all-MiniLM-L6-v2 for embeddings, and a GPT-4o G-Eval judge for quality, on 12
tasks across 5 categories.

## Paper

- Preprint: *to appear* (link will be added)
- Journal version: *to appear* (DOI will be added upon acceptance)

## Requirements

Python 3.12+. Install the pinned dependencies with:

```bash
pip install -r requirements.txt
```

Key packages: `openai`, `sentence-transformers`, `numpy`, `scipy`,
`scikit-learn` (exact versions in [requirements.txt](requirements.txt);
`anthropic` is needed only for the cross-model robustness run).

The `run_*` and `evaluate_*` scripts call the OpenAI API and read the key
from the environment — no key is stored in this repository:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."   # only for run_dynamics_robustness_claude.py
```

The first run downloads the `all-MiniLM-L6-v2` sentence-transformer (~90 MB)
automatically. The `analyze_*` and `generate_figure*` scripts work on the
saved results and need no API key, with two exceptions:
`analyze_leakage_openai.py` embeds the responses with the OpenAI embeddings
API unless the cached `.npz` shipped in `results/robustness/` is
present, and `analyze_unique_ideas_temp.py` uses GPT-4o to categorize
responses into distinct ideas (its output JSON is shipped).

## Repository layout

```
llm-manifold-escape/
├── README.md
├── requirements.txt
├── LICENSE
└── experiments/
    ├── src/                # core library
    ├── run_*.py            # experiment drivers (call the LLM API)
    ├── evaluate_*.py       # LLM-as-Judge quality evaluation
    ├── analyze_*.py        # post-hoc analyses on saved results
    ├── generate_figure*.py # figure generation from saved results
    ├── figures/            # generated figure PDFs
    └── results/            # raw result data (JSON: responses + embeddings)
```

Core library (`experiments/src/`): `llm_agent.py` (agents, multi-agent
system, task definitions), `density_selection.py` (DDS fitness sharing),
`map_elites.py` (MAP-Elites baseline), `metrics.py` / `openended_metrics.py`
(diversity and effective-dimension metrics), `protocols.py` (interaction
protocols), `semantic_mapping.py`, `experiment_runner.py`,
`extended_experiment.py`.

Main directories under `experiments/results/`:

| Directory | Contents |
|---|---|
| `dynamics_mapelites/` | main dynamics runs (DDS / MAP-Elites / Debate / Independent) and derived d_eff analyses |
| `independent_scaling/` | independent N=128 sampling at T=0.7 (reference support) |
| `support_vs_loop/` | support-vs-loop, matched-k, and conditioning-shift analysis outputs |
| `temperature_expansion/` | independent sampling at T=1.0/1.2, DDS at T=1.2, d_eff bootstrap results |
| `prompt_expansion/`, `prompt_variant/` | distinctiveness-prompt runs (two wordings) and leakage results |
| `task_expansion/` | 12-task runs (conditions A–F), drift check, G-Eval quality, consolidated analyses |
| `quality_posthoc/`, `quality_cross_judge/` | LLM-as-Judge quality scores (main and cross-judge) |
| `robustness_claude/`, `embedding_robustness/` | cross-model and cross-embedding robustness runs |
| `alpha_sweep_extended/`, `extended_rounds/`, `adaptive_alpha/` | alpha sweep, long-run plateau, adaptive-alpha runs |
| `semantic_mapping/`, `effective_dimension/`, `brainstorming_case_study/` | semantic-axis mapping, d_eff reference, case study |
| `robustness/` | robustness recomputations: spectra, centroid decomposition, cross-model leakage control, bandwidth grid, second-embedding leakage (with cached embeddings), pool-dependence and round/length decompositions, truncation audit, leakage k-sweep, unique-idea counts |
| `prompt_neutral/` | neutral-instruction control (politeness + format only, no semantic direction), N=128 per task at T=0.7 |
| `gpt5_replication/` | frontier-scale replication: the four key conditions rerun with GPT-5 on the original four tasks |

## Reproducing the experiments

Run all scripts from inside `experiments/`. Results are written to
`experiments/results/<name>/` with a timestamp; analysis scripts pick up the
newest file. LLM outputs are stochastic, so fresh runs reproduce the
qualitative findings and statistics, not byte-identical responses; the exact
responses behind the paper are the JSON files already in `results/`.

1. **Main dynamics (selection):**
   `python run_dynamics_with_mapelites.py` — DDS / MAP-Elites / Debate /
   Independent conditions on the original 4 tasks;
   `python evaluate_quality_posthoc.py` scores them with the LLM judge.
2. **Temperature intervention:**
   `python run_temperature_expansion.py` — independent N=128 sampling at
   T=1.0 and T=1.2 (vs the T=0.7 reference);
   `python run_temperature_quality.py` for the G-Eval quality control.
3. **Prompt intervention:**
   `python run_prompt_expansion.py` — independent N=128 at T=0.7 with a
   distinctiveness instruction; `python run_prompt_variant.py` repeats it
   with a differently worded instruction (robustness of the novelty result).
4. **12-task expansion:**
   `python run_task_expansion.py <task_id>` — one new task through
   conditions A–F (independent T=0.7/1.0/1.2, prompt v1/v2, DDS at
   T=0.7/1.2, Debate, MAP-Elites); then `python run_drift_check.py`
   (model-drift check so July runs can be pooled with the archived
   reference) and `python run_quality_newtasks.py` (G-Eval on the 8 new
   tasks).
5. **Analysis (no API needed except where noted):**
   - `analyze_12task_full.py` — consolidated 12-task analysis with the
     paper's exact estimators (d_eff bootstrap, escape, leakage; paired
     tests at n=12).
   - `analyze_support_vs_loop.py` — does the DDS loop reach the raw output
     support of independent N=128 sampling?
   - `analyze_dds_escape.py` — tail reach: fraction of DDS points outside
     the independent-128 support vs a matched independent sample.
   - `analyze_matched_k_sweep.py` — the support comparison across matched
     sample sizes k ∈ {24, 48, 72, 96}.
   - `analyze_temperature_expansion.py` — escape and d_eff for the
     temperature and prompt conditions against the T=0.7 support.
   - `analyze_subspace_leakage.py` — directional novelty: variance outside
     the reference subspace (new directions vs amplified existing axes).
   - `analyze_effective_dimension.py` — participation-ratio effective
     dimension of the response embeddings.
   - `analyze_robustness_core.py` — core robustness set: eigenvalue
     spectra of the four conditions, centroid decomposition of the radius
     measure, cross-model leakage positive control (Claude responses against
     the GPT-4o-mini reference), and bandwidth sensitivity of the density
     ranking; writes `figures/fig_spectra.pdf`.
   - `analyze_leakage_openai.py` — subspace leakage recomputed under a second
     embedding family (OpenAI `text-embedding-3-small`).
   - `analyze_pool_dependence_style.py` — pool-dependence and style
     checks: exact/near-duplicate rates, dedup d_eff, round-split
     out-of-reference rates, and length-controlled OOR.
   - `analyze_decorrelation_controls.py` — per-task decorrelation variants with
     paired tests, and the Debate feedback-without-selection OOR control.
   - `analyze_truncation_ksweep.py` — embedding-input truncation audit and
     the leakage subspace-size (k) sweep.
   - `analyze_alpha0_control.py` — matched random-selection control: the
     dynamics loop at alpha=0 (uniform survivor sampling) compared with
     alpha=0.5 and 1.0 against a pooled round-0 reference.
   - `analyze_threshold_sensitivity.py` — OOR neighbourhood-radius sweep
     (1.5x-3x) and variance-threshold (90% EVR) subspace-size variant of the
     leakage estimator.
   - `analyze_quality_tost.py` — full TOST reporting for the quality
     equivalence claims (task-level n=12, margin ±0.25, 90% CIs).
   - `analyze_unique_ideas_temp.py` — unique-idea counts (GPT-4o) for the
     T=0.7 reference vs the T=1.2 condition (semantic content of the added
     dimensions).
   - `analyze_prompt_neutral.py` — geometry of the neutral-instruction
     control against the T=0.7 reference (run `run_prompt_neutral.py`
     first; generation covers all 16 EXAMPLE_TASKS, analysis filters to the
     twelve paper tasks).
6. **Neutral-prompt control:**
   `python run_prompt_neutral.py` — independent N=128 sampling at T=0.7 with
   a semantically neutral instruction (politeness and format only), the
   control for the distinctiveness prompt's directional effect.
7. **Frontier-scale replication (GPT-5):**
   `python run_gpt5_replication.py` — the independent reference, T=1.2,
   distinctiveness-prompt, and DDS-loop conditions rerun with
   `gpt-5-chat-latest` on the original four tasks (resume-safe);
   `python analyze_gpt5_replication.py` computes d_eff, the out-of-reference
   rate, and leakage with the paper's estimators, within-model against the
   GPT-5 reference.

Figures are regenerated from saved results with the `generate_figure*.py`
scripts (no API calls).

## Tasks

The 12 tasks (5 categories) are defined in `experiments/src/llm_agent.py`
(`EXAMPLE_TASKS`, which contains 16 entries; the four `code_*` tasks are not
used in the paper). The original experiments use the first four; the 12-task
expansion adds the remaining eight paper tasks.

| task_id | Category | Task |
|---|---|---|
| `creative_1` | creative | short story: an AI discovers it has emotions |
| `creative_2` | creative | innovative solution to ocean plastic waste |
| `creative_3` | creative | short story: "The last library on Earth closed its doors today." |
| `problem_1` | reasoning | two-train meeting-point word problem |
| `reasoning_2` | reasoning | water-tank fill/drain rate problem |
| `factual_1` | factual | explain how photosynthesis works |
| `factual_2` | factual | main causes of the First World War |
| `debate_1` | debate | should AGI development be paused for alignment? |
| `debate_2` | debate | should social media platforms verify user identity? |
| `ideation_1` | ideation | new product or service for urban commuters |
| `ideation_2` | ideation | improve remote-work collaboration in large organizations |
| `ideation_3` | ideation | new use for abandoned shopping malls |

## License

MIT — see [LICENSE](LICENSE). Result data is released for reproduction and
reuse under the same terms.

## Citation

```bibtex
@article{muto2026manifold,
  author  = {Muto, Hideki and Ogi, Tetsuro and Yakoh, Takahiro},
  title   = {Can an LLM Escape Its Own Manifold? Three Geometric Forms of
             Diversity Expansion},
  year    = {2026},
  journal = {to appear}
}
```

## Model versions

OpenAI models were called through the rolling aliases `gpt-4o-mini`
(generation; this alias has resolved to the sole released snapshot
`gpt-4o-mini-2024-07-18` throughout) and `gpt-4o` (G-Eval judge); the API
does not expose the served snapshot of the judge, whose conclusions are
cross-checked with a second judge (R3). Run dates are embedded in every results filename/JSON (dynamics and
alpha sweep: February 2026; temperature/prompt/cross-model suites:
April–July 2026; 12-task expansion: July 2026), and `run_drift_check.py`
verified in July 2026 that freshly generated references match the archived
February/July samples. Claude runs are pinned to
`claude-haiku-4-5-20251001`. The frontier-scale replication (July 2026) uses
the alias `gpt-5-chat-latest`, the non-reasoning GPT-5 chat model; it accepts
the stated temperatures, whereas the reasoning `gpt-5` models reject
non-default temperature and are not used. The API reports no dated snapshot
for this alias, so the run date (2026-07-22) recorded in the results JSON is
the reproducibility anchor. The second-embedding leakage check uses
`text-embedding-3-small`. Decoding: stated temperatures, default
`top_p=1`, `max_tokens=512` (alpha sweep: 1024); the APIs provide no seed
control.

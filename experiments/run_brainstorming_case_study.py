#!/usr/bin/env python3
"""
Brainstorming Case Study: Practical Application of DDS

Demonstrates DDS on realistic brainstorming tasks where idea diversity
has clear practical value. Measures:
1. Semantic diversity (cosine distance)
2. Unique idea count (LLM-based categorization)
3. Quality (G-Eval with GPT-4o judge)

3 Conditions: DDS alpha=0.5, MAP-Elites, Independent
3 Brainstorming tasks, 5 trials each, 8 agents, 3 rounds

Usage:
    python run_brainstorming_case_study.py [--yes]
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_agent import MultiAgentSystem, AgentConfig, Task
from src.protocols import DDSProtocol, DDSConfig, IndependentProtocol, ProtocolConfig
from src.density_selection import AgentResponse
from src.map_elites import MAPElitesSelector, MAPElitesConfig


# ============================================================
# Brainstorming Tasks
# ============================================================
BRAINSTORM_TASKS = [
    Task(
        task_id="brainstorm_urban",
        prompt=(
            "Brainstorm creative and practical ideas for making a mid-sized city "
            "(population ~500,000) more environmentally sustainable within the next "
            "10 years. Consider infrastructure, policy, community engagement, "
            "technology, and economic incentives. Propose 3-5 distinct ideas."
        ),
        category="brainstorming"
    ),
    Task(
        task_id="brainstorm_remote",
        prompt=(
            "Brainstorm innovative solutions to combat loneliness and social "
            "isolation among remote workers. Consider technology tools, workplace "
            "policies, community spaces, social rituals, and mental health support. "
            "Propose 3-5 distinct ideas."
        ),
        category="brainstorming"
    ),
    Task(
        task_id="brainstorm_food",
        prompt=(
            "Brainstorm creative ways to reduce food waste at the consumer level "
            "(households and restaurants). Consider technology, behavioral nudges, "
            "business models, education, and community programs. "
            "Propose 3-5 distinct ideas."
        ),
        category="brainstorming"
    ),
]

# Task prompts for G-Eval (matching task_id to prompt)
TASK_PROMPTS = {t.task_id: t.prompt for t in BRAINSTORM_TASKS}


# ============================================================
# Unique Idea Counting (LLM-based)
# ============================================================
CATEGORIZE_PROMPT = """You are an expert at analyzing brainstorming outputs.

## Task
{task_prompt}

## All Responses
{all_responses}

## Instructions
1. Read all the responses above carefully.
2. Identify every DISTINCT idea proposed across all responses.
3. Two ideas are "distinct" if they address fundamentally different approaches, mechanisms, or domains. Minor variations of the same core idea count as ONE idea.
4. List each unique idea as a short label (3-8 words).

## Output Format
Respond in exactly this JSON format (no other text):
{{"unique_ideas": ["idea label 1", "idea label 2", ...], "count": <number>}}"""


def count_unique_ideas(client, responses_texts, task_prompt, model="gpt-4o"):
    """Count unique ideas across all responses using LLM categorization."""
    all_text = "\n\n---\n\n".join(
        f"Response {i+1}:\n{t}" for i, t in enumerate(responses_texts)
    )

    prompt = CATEGORIZE_PROMPT.format(
        task_prompt=task_prompt,
        all_responses=all_text
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2000,
    )

    raw = completion.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0] if "```" in raw else raw
        raw = raw.strip()

    result = json.loads(raw)
    return result["unique_ideas"], result["count"]


# ============================================================
# G-Eval Quality Evaluation
# ============================================================
GEVAL_PROMPT = """You are an expert evaluator assessing the quality of a response to a given task.

## Task
{task_prompt}

## Response to Evaluate
{response}

## Evaluation Instructions
Evaluate the response on three dimensions using the rubrics below.
First, briefly reason about each dimension (1-2 sentences), then assign a score.

### Dimension 1: Coherence (1-5)
How well-structured and logically organized is the response?
1 - Incoherent, disjointed, or contradictory
2 - Poorly organized with significant logical gaps
3 - Adequate structure but some unclear transitions
4 - Well-organized with clear logical flow
5 - Exceptionally coherent with seamless logical progression

### Dimension 2: Relevance (1-5)
How well does the response address the specific task requirements?
1 - Completely off-topic or ignores the task
2 - Partially addresses the task with major omissions
3 - Addresses the main task requirements adequately
4 - Thoroughly addresses the task with good coverage
5 - Comprehensively addresses all aspects of the task

### Dimension 3: Depth (1-5)
How insightful, detailed, and substantive is the response?
1 - Superficial with no meaningful analysis
2 - Limited depth with only surface-level treatment
3 - Moderate depth with some substantive points
4 - Good depth with well-developed analysis
5 - Exceptional depth with nuanced, insightful analysis

## Output Format
Respond in exactly this JSON format (no other text):
{{"coherence": <1-5>, "relevance": <1-5>, "depth": <1-5>}}"""


def evaluate_quality(client, response_text, task_prompt, model="gpt-4o"):
    """Evaluate a single response using G-Eval."""
    prompt = GEVAL_PROMPT.format(
        task_prompt=task_prompt,
        response=response_text
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )

    raw = completion.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0] if "```" in raw else raw
        raw = raw.strip()

    scores = json.loads(raw)
    return {
        "coherence": int(scores["coherence"]),
        "relevance": int(scores["relevance"]),
        "depth": int(scores["depth"]),
    }


# ============================================================
# Diversity & Text Helpers
# ============================================================
def compute_diversity(responses):
    """Compute mean pairwise cosine distance from Response objects."""
    embeddings = np.array([r.embedding for r in responses])
    if len(embeddings) < 2:
        return 0.0
    from scipy.spatial.distance import cdist
    dists = cdist(embeddings, embeddings, metric='cosine')
    n = len(embeddings)
    upper = dists[np.triu_indices(n, k=1)]
    return float(np.mean(upper))


def extract_texts(responses):
    """Extract response texts from Response objects."""
    return [{"agent_id": r.agent_id, "text": r.text} for r in responses]


# ============================================================
# Run Conditions
# ============================================================
def run_dds_condition(system, task, n_rounds, n_survive):
    """Run DDS alpha=0.5."""
    config = DDSConfig(
        n_rounds=n_rounds,
        n_agents=system.n_agents,
        alpha=0.5,
        beta=2.0,
        bandwidth=0.3,
        n_survive=n_survive
    )
    protocol = DDSProtocol(config)
    result = protocol.run(system, task, quality_evaluator=None)

    round_diversities = []
    response_texts = []
    for round_responses in result["round_history"]:
        round_diversities.append(compute_diversity(round_responses))
        response_texts.append(extract_texts(round_responses))

    return {
        "round_diversities": round_diversities,
        "final_diversity": round_diversities[-1],
        "n_rounds": len(round_diversities),
        "response_texts": response_texts,
    }


def run_mapelites_condition(system, task, n_rounds):
    """Run MAP-Elites."""
    me_config = MAPElitesConfig(n_bins_per_dim=5, n_behavior_dims=2)
    selector = MAPElitesSelector(me_config, behavior_method="pca")

    system.reset_all()

    round_diversities = []
    response_texts = []
    archive_coverages = []

    for round_idx in range(n_rounds):
        if round_idx == 0:
            responses = system.generate_responses(task)
        else:
            elites = selector.archive.get_all_elites()

            def context_provider(agent_id, _elites=elites):
                elite_texts = [
                    f"Selected response: {e.response_text}"
                    for e in _elites
                ]
                return "\n\n".join(elite_texts)

            me_task = Task(
                task_id=f"{task.task_id}_round{round_idx}",
                prompt=f"{task.prompt}\n\nBuild upon or differentiate from the context.",
                category=task.category
            )
            responses = system.generate_responses(me_task, context_provider)

        div = compute_diversity(responses)
        round_diversities.append(div)
        response_texts.append(extract_texts(responses))

        agent_responses = [
            AgentResponse(
                agent_id=r.agent_id,
                response_text=r.text,
                embedding=r.embedding,
                quality_score=0.75,
                generation=round_idx
            )
            for r in responses
        ]
        stats = selector.update_archive(agent_responses)
        archive_coverages.append(stats["coverage"])

    return {
        "round_diversities": round_diversities,
        "final_diversity": round_diversities[-1],
        "n_rounds": len(round_diversities),
        "response_texts": response_texts,
        "archive_coverage": archive_coverages,
    }


def run_independent_condition(system, task):
    """Run Independent (single round, no interaction)."""
    config = ProtocolConfig(n_rounds=1, n_agents=system.n_agents)
    protocol = IndependentProtocol(config)
    result = protocol.run(system, task, quality_evaluator=None)

    responses = result["final_responses"]
    div = compute_diversity(responses)

    return {
        "round_diversities": [div],
        "final_diversity": div,
        "n_rounds": 1,
        "response_texts": [extract_texts(responses)],
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("BRAINSTORMING CASE STUDY")
    print("Practical Application of Density-Dependent Selection")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print()

    # Configuration
    N_AGENTS = 8
    N_ROUNDS = 3
    N_SURVIVE = 5
    N_TRIALS = 5
    BACKEND = "openai"
    MODEL = "gpt-4o-mini"

    tasks = BRAINSTORM_TASKS

    # Cost estimation
    gen_calls = 3 * len(tasks) * N_TRIALS * N_AGENTS * N_ROUNDS  # 3 conds × 3 tasks × 5 trials × 8 agents × 3 rounds
    # Independent only has 1 round
    gen_calls -= len(tasks) * N_TRIALS * N_AGENTS * (N_ROUNDS - 1)

    print("Configuration:")
    print(f"  Backend: {BACKEND}")
    print(f"  Model: {MODEL}")
    print(f"  N_agents: {N_AGENTS}")
    print(f"  N_rounds: {N_ROUNDS}")
    print(f"  N_survive (DDS): {N_SURVIVE}")
    print(f"  N_trials per task: {N_TRIALS}")
    print(f"  Tasks: {[t.task_id for t in tasks]}")
    print(f"  Conditions: DDS alpha=0.5, MAP-Elites, Independent")
    print()
    print(f"Estimated generation calls: {gen_calls:,}")
    print(f"Estimated generation cost: ~${gen_calls * 0.0002:.2f} USD")
    print(f"Estimated generation time: ~{gen_calls * 2 / 60:.0f} minutes")
    print()

    if "--yes" not in sys.argv:
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Initialize system
    agent_config = AgentConfig(
        agent_id=0,
        backend=BACKEND,
        model=MODEL,
        temperature=0.7,
        max_tokens=1024,
        system_prompt="You are a creative brainstorming assistant. Generate diverse, practical, and innovative ideas."
    )
    system = MultiAgentSystem(n_agents=N_AGENTS, agent_config_template=agent_config)

    # Storage
    all_results = {
        "config": {
            "n_agents": N_AGENTS,
            "n_rounds": N_ROUNDS,
            "n_survive": N_SURVIVE,
            "n_trials": N_TRIALS,
            "backend": BACKEND,
            "model": MODEL,
            "tasks": [t.task_id for t in tasks],
            "conditions": ["dds_alpha_0.5", "map_elites", "independent"],
        },
        "timestamp": datetime.now().isoformat(),
        "conditions": {},
    }

    # ============================================================
    # Phase 1: Generation
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: GENERATION")
    print("=" * 70)

    conditions = [
        ("dds_alpha_0.5", lambda s, t: run_dds_condition(s, t, N_ROUNDS, N_SURVIVE)),
        ("map_elites", lambda s, t: run_mapelites_condition(s, t, N_ROUNDS)),
        ("independent", lambda s, t: run_independent_condition(s, t)),
    ]

    for cond_name, run_fn in conditions:
        print(f"\n{'=' * 60}")
        print(f"Condition: {cond_name}")
        print(f"{'=' * 60}")

        condition_results = []

        for task in tasks:
            task_trials = []

            for trial_idx in range(N_TRIALS):
                system.reset_all()
                print(f"  [{task.task_id}] Trial {trial_idx+1}/{N_TRIALS}...",
                      end="", flush=True)
                try:
                    result = run_fn(system, task)
                    task_trials.append(result)
                    print(f" div={result['final_diversity']:.4f}")
                except Exception as e:
                    print(f" ERROR: {e}")
                    task_trials.append(None)

            condition_results.append({
                "task_id": task.task_id,
                "trials": [t for t in task_trials if t is not None],
            })

        all_results["conditions"][cond_name] = condition_results

    # Save generation results
    output_dir = Path("results/brainstorming_case_study")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"brainstorming_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nGeneration results saved: {output_file}")

    # ============================================================
    # Phase 2: Unique Idea Counting + G-Eval Quality
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: UNIQUE IDEA COUNTING + G-EVAL QUALITY")
    print("=" * 70)

    from openai import OpenAI
    client = OpenAI()

    eval_count = 0
    idea_count_results = {}
    quality_results = {}

    for cond_name in ["dds_alpha_0.5", "map_elites", "independent"]:
        print(f"\n--- {cond_name} ---")
        cond_idea_counts = []
        cond_quality = []

        for task_data in all_results["conditions"][cond_name]:
            task_id = task_data["task_id"]
            task_prompt = TASK_PROMPTS[task_id]

            for trial_idx, trial in enumerate(task_data["trials"]):
                # Collect ALL response texts across rounds
                all_texts = []
                for round_texts in trial["response_texts"]:
                    for resp in round_texts:
                        all_texts.append(resp["text"])

                # Count unique ideas
                try:
                    ideas, n_ideas = count_unique_ideas(
                        client, all_texts, task_prompt
                    )
                    cond_idea_counts.append({
                        "task_id": task_id,
                        "trial_idx": trial_idx,
                        "unique_ideas": ideas,
                        "n_unique_ideas": n_ideas,
                        "n_responses": len(all_texts),
                    })
                    print(f"  [{task_id}] Trial {trial_idx+1}: "
                          f"{n_ideas} unique ideas from {len(all_texts)} responses")
                except Exception as e:
                    print(f"  [{task_id}] Trial {trial_idx+1}: IDEA COUNT ERROR: {e}")
                    cond_idea_counts.append({
                        "task_id": task_id,
                        "trial_idx": trial_idx,
                        "unique_ideas": [],
                        "n_unique_ideas": 0,
                        "n_responses": len(all_texts),
                        "error": str(e),
                    })

                # G-Eval quality for each response
                trial_qualities = []
                for round_idx, round_texts in enumerate(trial["response_texts"]):
                    round_qualities = []
                    for resp in round_texts:
                        try:
                            scores = evaluate_quality(
                                client, resp["text"], task_prompt
                            )
                            round_qualities.append(scores)
                            eval_count += 1
                        except Exception as e:
                            round_qualities.append(None)
                            eval_count += 1

                        if eval_count % 50 == 0:
                            time.sleep(1)

                    trial_qualities.append(round_qualities)

                cond_quality.append({
                    "task_id": task_id,
                    "trial_idx": trial_idx,
                    "round_qualities": trial_qualities,
                })

                # Print quality summary
                valid = [s for rq in trial_qualities for s in rq if s is not None]
                if valid:
                    mean_c = np.mean([s["coherence"] for s in valid])
                    mean_r = np.mean([s["relevance"] for s in valid])
                    mean_d = np.mean([s["depth"] for s in valid])
                    print(f"           quality: C={mean_c:.1f} R={mean_r:.1f} D={mean_d:.1f}")

        idea_count_results[cond_name] = cond_idea_counts
        quality_results[cond_name] = cond_quality

    # Add evaluation results
    all_results["idea_counts"] = idea_count_results
    all_results["quality_geval"] = quality_results
    all_results["eval_stats"] = {
        "total_quality_evals": eval_count,
        "quality_model": "gpt-4o",
        "idea_count_model": "gpt-4o",
    }

    # Save final results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nFull results saved: {output_file}")

    # ============================================================
    # Phase 3: Summary Statistics
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    from scipy import stats

    for cond_name in ["dds_alpha_0.5", "map_elites", "independent"]:
        cond_data = all_results["conditions"][cond_name]

        # Diversity
        all_final_divs = []
        for task_data in cond_data:
            for trial in task_data["trials"]:
                all_final_divs.append(trial["final_diversity"])

        # Unique ideas
        idea_counts = [
            ic["n_unique_ideas"]
            for ic in idea_count_results[cond_name]
            if ic["n_unique_ideas"] > 0
        ]

        # Quality
        all_quals = []
        for entry in quality_results[cond_name]:
            scores = [
                s for rq in entry["round_qualities"] for s in rq if s is not None
            ]
            if scores:
                overall = np.mean([(s["coherence"] + s["relevance"] + s["depth"]) / 3
                                   for s in scores])
                all_quals.append(overall)

        print(f"\n{cond_name}:")
        print(f"  Diversity: {np.mean(all_final_divs):.4f} +/- {np.std(all_final_divs, ddof=1):.4f}")
        if idea_counts:
            print(f"  Unique ideas: {np.mean(idea_counts):.1f} +/- {np.std(idea_counts, ddof=1):.1f}")
        if all_quals:
            print(f"  Quality (G-Eval): {np.mean(all_quals):.3f} +/- {np.std(all_quals, ddof=1):.3f}")

    # Pairwise comparisons
    print("\n--- Statistical Tests ---")
    for metric_name, get_vals_fn in [
        ("Diversity", lambda cn: [
            trial["final_diversity"]
            for td in all_results["conditions"][cn]
            for trial in td["trials"]
        ]),
        ("Unique Ideas", lambda cn: [
            ic["n_unique_ideas"]
            for ic in idea_count_results[cn]
            if ic["n_unique_ideas"] > 0
        ]),
    ]:
        print(f"\n  {metric_name}:")
        for a, b in [("dds_alpha_0.5", "independent"), ("map_elites", "independent"),
                     ("dds_alpha_0.5", "map_elites")]:
            va = get_vals_fn(a)
            vb = get_vals_fn(b)
            n = min(len(va), len(vb))
            if n < 3:
                continue
            t_stat, p_val = stats.ttest_rel(va[:n], vb[:n])
            diff = np.array(va[:n]) - np.array(vb[:n])
            d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            print(f"    {a} vs {b}: t({n-1})={t_stat:.3f}, p={p_val:.4f}, d={d:.3f} ({sig})")

    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

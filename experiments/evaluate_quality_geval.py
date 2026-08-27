#!/usr/bin/env python3
"""
G-Eval Style Multi-Dimensional Quality Evaluation

Evaluates response quality using LLM-as-Judge with explicit rubrics
for three dimensions: Coherence, Relevance, Depth.

Based on G-Eval framework (Liu et al., 2023):
- Detailed per-dimension rubrics (1-5 scale)
- Chain-of-thought evaluation
- Structured output

Originality is deliberately excluded to avoid double-rewarding diversity.

Uses GPT-4o as judge (different from GPT-4o-mini used for generation)
to avoid self-evaluation bias.

Usage:
    python evaluate_quality_geval.py [--yes]
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Task definitions (matching EXAMPLE_TASKS)
TASK_PROMPTS = {
    "creative_1": "Write a short story about an AI that discovers it has emotions.",
    "creative_2": "Describe an innovative solution to reduce plastic waste in oceans.",
    "problem_1": "A train travels from City A to City B at 60 mph. Another train leaves City B 30 minutes later traveling toward City A at 80 mph. If the cities are 280 miles apart, how far from City A will they meet?",
    "debate_1": "Should artificial general intelligence (AGI) development be paused until we have better alignment techniques? Provide arguments for your position.",
}

# G-Eval style evaluation prompt with detailed rubrics
GEVAL_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of a response to a given task.

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


def evaluate_single(client, response_text, task_prompt, model="gpt-4o"):
    """Evaluate a single response using G-Eval style prompt."""
    prompt = GEVAL_PROMPT_TEMPLATE.format(
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

    # Parse JSON response
    # Handle cases where model wraps in ```json ... ```
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


def main():
    # Locate data file
    results_dir = Path("results/dynamics_mapelites")
    files = sorted(results_dir.glob("dynamics_mapelites_*.json"))
    if not files:
        print("No results files found in results/dynamics_mapelites/")
        return
    input_file = files[-1]

    with open(input_file) as f:
        data = json.load(f)

    # Conditions to evaluate (exclude debate)
    conditions_to_eval = ['dds_alpha_0.0', 'dds_alpha_0.5', 'dds_alpha_1.0',
                          'map_elites', 'independent']

    # Count total evaluations
    total_evals = 0
    for cond_name in conditions_to_eval:
        for task_data in data["conditions"][cond_name]:
            for trial in task_data["trials"]:
                if "response_texts" in trial:
                    for round_texts in trial["response_texts"]:
                        total_evals += len(round_texts)

    print("=" * 70)
    print("G-EVAL STYLE MULTI-DIMENSIONAL QUALITY EVALUATION")
    print("=" * 70)
    print(f"Input: {input_file}")
    print(f"Conditions: {conditions_to_eval}")
    print(f"Total evaluations: {total_evals:,}")
    print(f"Dimensions: coherence, relevance, depth (1-5 scale each)")
    print(f"Estimated cost: ~${total_evals * 0.002:.2f} USD (GPT-4o)")
    print(f"Estimated time: ~{total_evals * 1.5 / 60:.0f} minutes")
    print(f"Start time: {datetime.now().isoformat()}")
    print()

    if "--yes" not in sys.argv:
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Initialize OpenAI client
    from openai import OpenAI
    client = OpenAI()

    eval_count = 0
    error_count = 0
    quality_results = {}

    for cond_name in conditions_to_eval:
        print(f"\n{'=' * 60}")
        print(f"Condition: {cond_name}")
        print(f"{'=' * 60}")

        cond_qualities = []

        for task_data in data["conditions"][cond_name]:
            task_id = task_data["task_id"]
            task_prompt = TASK_PROMPTS.get(task_id, "Unknown task")

            for trial_idx, trial in enumerate(task_data["trials"]):
                if "response_texts" not in trial:
                    continue

                trial_qualities = []
                for round_idx, round_texts in enumerate(trial["response_texts"]):
                    round_qualities = []
                    for resp in round_texts:
                        try:
                            scores = evaluate_single(
                                client, resp["text"], task_prompt
                            )
                            round_qualities.append(scores)
                            eval_count += 1
                        except Exception as e:
                            print(f"    ERROR [{task_id} T{trial_idx+1} "
                                  f"R{round_idx} A{resp['agent_id']}]: {e}")
                            round_qualities.append(None)
                            error_count += 1

                        # Rate limiting
                        if eval_count % 50 == 0:
                            time.sleep(1)

                    trial_qualities.append(round_qualities)

                cond_qualities.append({
                    "task_id": task_id,
                    "trial_idx": trial_idx,
                    "round_qualities": trial_qualities,
                })

                # Progress
                valid = [s for rq in trial_qualities for s in rq if s is not None]
                if valid:
                    mean_c = np.mean([s["coherence"] for s in valid])
                    mean_r = np.mean([s["relevance"] for s in valid])
                    mean_d = np.mean([s["depth"] for s in valid])
                    print(f"  [{task_id}] Trial {trial_idx+1}: "
                          f"C={mean_c:.2f} R={mean_r:.2f} D={mean_d:.2f} "
                          f"({eval_count}/{total_evals})")

        quality_results[cond_name] = cond_qualities

    # Save results
    output_dir = Path("results/quality_geval")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"quality_geval_{timestamp}.json"

    output = {
        "source_file": str(input_file),
        "timestamp": datetime.now().isoformat(),
        "evaluator": {
            "method": "geval_multi_dimensional",
            "model": "gpt-4o",
            "generation_model": "gpt-4o-mini",
            "temperature": 0.0,
            "dimensions": ["coherence", "relevance", "depth"],
            "scale": "1-5 per dimension",
            "framework": "G-Eval (Liu et al., 2023) adapted",
            "note": "Originality excluded to avoid double-rewarding diversity. "
                    "Judge model (GPT-4o) differs from generation model (GPT-4o-mini) "
                    "to avoid self-evaluation bias.",
        },
        "stats": {
            "total_evaluations": eval_count,
            "errors": error_count,
        },
        "quality_results": quality_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 70}")

    # Print summary statistics
    print(f"\n{'=' * 70}")
    print("QUALITY SUMMARY (G-Eval)")
    print(f"{'=' * 70}\n")

    from scipy import stats as scipy_stats

    condition_scores = {}
    for cond_name in conditions_to_eval:
        cond_data = quality_results[cond_name]
        trial_means = {"coherence": [], "relevance": [], "depth": [], "overall": []}

        for entry in cond_data:
            all_scores = []
            for rq in entry["round_qualities"]:
                for s in rq:
                    if s is not None:
                        all_scores.append(s)

            if all_scores:
                c = np.mean([s["coherence"] for s in all_scores])
                r = np.mean([s["relevance"] for s in all_scores])
                d = np.mean([s["depth"] for s in all_scores])
                trial_means["coherence"].append(c)
                trial_means["relevance"].append(r)
                trial_means["depth"].append(d)
                trial_means["overall"].append((c + r + d) / 3)

        condition_scores[cond_name] = trial_means

        print(f"{cond_name}:")
        for dim in ["coherence", "relevance", "depth", "overall"]:
            vals = trial_means[dim]
            print(f"  {dim:12s}: {np.mean(vals):.3f} +/- {np.std(vals, ddof=1):.3f}")
        print()

    # Pairwise comparisons (overall quality)
    print("Pairwise comparisons (overall quality, paired t-test):")
    print("-" * 60)

    pairs = [
        ('dds_alpha_0.5', 'independent'),
        ('dds_alpha_0.0', 'independent'),
        ('dds_alpha_1.0', 'independent'),
        ('map_elites', 'independent'),
        ('dds_alpha_0.5', 'map_elites'),
    ]

    for a, b in pairs:
        va = condition_scores[a]["overall"]
        vb = condition_scores[b]["overall"]
        n = min(len(va), len(vb))
        t_stat, p_val = scipy_stats.ttest_rel(va[:n], vb[:n])
        diff = np.array(va[:n]) - np.array(vb[:n])
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"  {a} vs {b}: t({n-1})={t_stat:.3f}, p={p_val:.4f}, d={d:.3f} ({sig})")

    # Per-dimension pairwise (DDS 0.5 vs Independent)
    print(f"\nPer-dimension: DDS alpha=0.5 vs Independent:")
    print("-" * 60)
    for dim in ["coherence", "relevance", "depth"]:
        va = condition_scores["dds_alpha_0.5"][dim]
        vb = condition_scores["independent"][dim]
        n = min(len(va), len(vb))
        t_stat, p_val = scipy_stats.ttest_rel(va[:n], vb[:n])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"  {dim:12s}: DDS={np.mean(va[:n]):.3f}, Indep={np.mean(vb[:n]):.3f}, "
              f"t({n-1})={t_stat:.3f}, p={p_val:.4f} ({sig})")

    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

"""
Main Experiment Runner for Density-Dependent Selection

This script orchestrates the full experimental pipeline:
1. Load tasks and configuration
2. Run all protocols (Independent, Debate, DDS, etc.)
3. Compute metrics at each round
4. Save results and generate visualizations
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from .llm_agent import (
    MultiAgentSystem, AgentConfig, Task, Response,
    QualityEvaluator, EXAMPLE_TASKS
)
from .protocols import (
    Protocol, create_protocol,
    ProtocolConfig, DebateConfig, DiversityPromptConfig,
    RoleConfig, DDSConfig, FitnessSharingConfig
)
from .metrics import compute_all_metrics, PhaseTransitionAnalyzer


class ExperimentConfig:
    """Configuration for a full experiment run."""

    def __init__(
        self,
        name: str = "dds_experiment",
        n_agents: int = 5,
        n_rounds: int = 3,
        n_trials: int = 3,
        backend: str = "mock",  # "openai", "anthropic", "mock"
        model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        quality_eval_method: str = "random",  # "llm_judge", "random", "length_based"
        output_dir: str = "results",
        seed: int = 42
    ):
        self.name = name
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.n_trials = n_trials
        self.backend = backend
        self.model = model
        self.embedding_model = embedding_model
        self.quality_eval_method = quality_eval_method
        self.output_dir = output_dir
        self.seed = seed

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "n_agents": self.n_agents,
            "n_rounds": self.n_rounds,
            "n_trials": self.n_trials,
            "backend": self.backend,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "quality_eval_method": self.quality_eval_method,
            "seed": self.seed
        }


class ExperimentRunner:
    """
    Main experiment runner that orchestrates all protocols and metrics.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}

        # Set random seed
        np.random.seed(config.seed)

        # Initialize agent system
        agent_template = AgentConfig(
            agent_id=0,
            backend=config.backend,
            model=config.model,
            temperature=0.7
        )
        self.system = MultiAgentSystem(
            n_agents=config.n_agents,
            agent_config_template=agent_template,
            embedding_model=config.embedding_model
        )

        # Initialize quality evaluator
        self.quality_evaluator = QualityEvaluator(method=config.quality_eval_method)

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _evaluate_quality(self, text: str, task: Task) -> float:
        """Wrapper for quality evaluation."""
        return self.quality_evaluator.evaluate(text, task)

    def _compute_round_metrics(
        self,
        responses: List[Response]
    ) -> Dict:
        """Compute metrics for a set of responses."""
        embeddings = np.array([r.embedding for r in responses])
        texts = [r.text for r in responses]
        quality_scores = [r.quality_score for r in responses]

        return compute_all_metrics(embeddings, texts, quality_scores)

    def run_protocol(
        self,
        protocol: Protocol,
        task: Task,
        protocol_name: str
    ) -> Dict:
        """
        Run a single protocol on a task.

        Returns:
            Dictionary with protocol results and per-round metrics
        """
        # Run protocol
        result = protocol.run(
            self.system,
            task,
            quality_evaluator=self._evaluate_quality
        )

        # Compute metrics for each round
        round_metrics = []
        for round_responses in result["round_history"]:
            metrics = self._compute_round_metrics(round_responses)
            round_metrics.append(metrics)

        # Final metrics
        final_metrics = round_metrics[-1] if round_metrics else {}

        return {
            "protocol": protocol_name,
            "task_id": task.task_id,
            "task_category": task.category,
            "n_rounds": len(result["round_history"]),
            "round_metrics": round_metrics,
            "final_metrics": final_metrics,
            "protocol_config": result["metrics"],
            "selection_history": result.get("selection_history", [])
        }

    def run_comparison_experiment(
        self,
        tasks: List[Task],
        alpha_values: Optional[List[float]] = None
    ) -> Dict:
        """
        Run comparison experiment across all protocols.

        Args:
            tasks: List of tasks to evaluate
            alpha_values: List of α values to test for DDS (optional)

        Returns:
            Dictionary with all results
        """
        if alpha_values is None:
            alpha_values = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

        all_results = {
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "tasks": [{"task_id": t.task_id, "category": t.category, "prompt": t.prompt} for t in tasks],
            "protocol_results": {},
            "alpha_sweep": {}
        }

        # Define protocols to compare
        protocols_to_run = {
            "independent": ProtocolConfig(
                n_rounds=1,
                n_agents=self.config.n_agents
            ),
            "debate": DebateConfig(
                n_rounds=self.config.n_rounds,
                n_agents=self.config.n_agents
            ),
            "diversity_prompt": DiversityPromptConfig(
                n_rounds=self.config.n_rounds,
                n_agents=self.config.n_agents
            ),
            "role_assignment": RoleConfig(
                n_rounds=self.config.n_rounds,
                n_agents=self.config.n_agents
            ),
            "fitness_sharing": FitnessSharingConfig(
                n_rounds=self.config.n_rounds,
                n_agents=self.config.n_agents,
                share_radius=0.3
            )
        }

        # Run baseline protocols
        print("\n" + "=" * 60)
        print("Running Baseline Protocol Comparison")
        print("=" * 60)

        for protocol_name, protocol_config in tqdm(protocols_to_run.items(), desc="Protocols"):
            protocol = create_protocol(protocol_name, protocol_config)

            protocol_results = []
            for trial in range(self.config.n_trials):
                trial_results = []
                for task in tasks:
                    result = self.run_protocol(protocol, task, protocol_name)
                    trial_results.append(result)
                protocol_results.append(trial_results)

            all_results["protocol_results"][protocol_name] = protocol_results

        # Run DDS with different α values
        print("\n" + "=" * 60)
        print("Running DDS Alpha Sweep (Phase Transition Analysis)")
        print("=" * 60)

        for alpha in tqdm(alpha_values, desc="Alpha values"):
            dds_config = DDSConfig(
                n_rounds=self.config.n_rounds,
                n_agents=self.config.n_agents,
                alpha=alpha,
                beta=2.0,
                bandwidth=0.3,
                n_survive=max(2, self.config.n_agents // 2)
            )
            protocol = create_protocol("dds", dds_config)

            alpha_results = []
            for trial in range(self.config.n_trials):
                trial_results = []
                for task in tasks:
                    result = self.run_protocol(protocol, task, f"dds_alpha_{alpha}")
                    trial_results.append(result)
                alpha_results.append(trial_results)

            all_results["alpha_sweep"][str(alpha)] = alpha_results

        return all_results

    def analyze_phase_transition(
        self,
        results: Dict
    ) -> Dict:
        """
        Analyze phase transition from alpha sweep results.
        """
        alpha_sweep = results.get("alpha_sweep", {})
        if not alpha_sweep:
            return {}

        alpha_values = sorted([float(a) for a in alpha_sweep.keys()])

        # Aggregate niche counts for each alpha
        niche_counts = []
        diversity_means = []

        for alpha in alpha_values:
            alpha_results = alpha_sweep[str(alpha)]

            # Average across trials and tasks
            trial_niches = []
            trial_diversity = []

            for trial_results in alpha_results:
                for task_result in trial_results:
                    final_metrics = task_result.get("final_metrics", {})
                    niche_info = final_metrics.get("niche_hierarchical", {})
                    diversity_info = final_metrics.get("diversity", {})

                    trial_niches.append(niche_info.get("n_niches", 1))
                    trial_diversity.append(diversity_info.get("mean_distance", 0))

            niche_counts.append(np.mean(trial_niches))
            diversity_means.append(np.mean(trial_diversity))

        # Detect critical point
        analyzer = PhaseTransitionAnalyzer()
        critical_point = analyzer.detect_critical_point(alpha_values, niche_counts)

        # Fit scaling law if critical point found
        scaling_law = {}
        if critical_point.get("alpha_c") is not None:
            scaling_law = analyzer.fit_scaling_law(
                alpha_values, niche_counts, critical_point["alpha_c"]
            )

        return {
            "alpha_values": alpha_values,
            "niche_counts": niche_counts,
            "diversity_means": diversity_means,
            "critical_point": critical_point,
            "scaling_law": scaling_law
        }

    def summarize_results(self, results: Dict) -> Dict:
        """
        Create summary statistics across all protocols.
        """
        summary = {
            "protocol_comparison": {},
            "best_diversity_protocol": None,
            "best_quality_protocol": None
        }

        protocol_results = results.get("protocol_results", {})

        for protocol_name, trials in protocol_results.items():
            # Aggregate metrics
            all_diversity = []
            all_quality = []
            all_niches = []

            for trial_results in trials:
                for task_result in trial_results:
                    final_metrics = task_result.get("final_metrics", {})
                    diversity_info = final_metrics.get("diversity", {})
                    quality_info = final_metrics.get("quality", {})
                    niche_info = final_metrics.get("niche_hierarchical", {})

                    all_diversity.append(diversity_info.get("mean_distance", 0))
                    all_quality.append(quality_info.get("mean", 0))
                    all_niches.append(niche_info.get("n_niches", 1))

            summary["protocol_comparison"][protocol_name] = {
                "diversity_mean": float(np.mean(all_diversity)),
                "diversity_std": float(np.std(all_diversity)),
                "quality_mean": float(np.mean(all_quality)),
                "quality_std": float(np.std(all_quality)),
                "niche_mean": float(np.mean(all_niches)),
                "niche_std": float(np.std(all_niches))
            }

        # Add best DDS result
        alpha_sweep = results.get("alpha_sweep", {})
        best_dds_diversity = 0
        best_dds_alpha = None

        for alpha_str, trials in alpha_sweep.items():
            for trial_results in trials:
                for task_result in trial_results:
                    final_metrics = task_result.get("final_metrics", {})
                    diversity = final_metrics.get("diversity", {}).get("mean_distance", 0)
                    if diversity > best_dds_diversity:
                        best_dds_diversity = diversity
                        best_dds_alpha = float(alpha_str)

        if best_dds_alpha is not None:
            summary["best_dds"] = {
                "alpha": best_dds_alpha,
                "diversity": best_dds_diversity
            }

        # Find best protocols
        if summary["protocol_comparison"]:
            best_div = max(
                summary["protocol_comparison"].items(),
                key=lambda x: x[1]["diversity_mean"]
            )
            best_qual = max(
                summary["protocol_comparison"].items(),
                key=lambda x: x[1]["quality_mean"]
            )
            summary["best_diversity_protocol"] = best_div[0]
            summary["best_quality_protocol"] = best_qual[0]

        return summary

    def save_results(self, results: Dict, filename: Optional[str] = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_{self.config.name}_{timestamp}.json"

        filepath = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        results_serializable = convert_numpy(results)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {filepath}")
        return filepath


def run_quick_test():
    """
    Run a quick test with mock agents to verify the pipeline works.
    """
    print("=" * 60)
    print("Quick Test: Density-Dependent Selection Experiment")
    print("=" * 60)

    config = ExperimentConfig(
        name="quick_test",
        n_agents=5,
        n_rounds=2,
        n_trials=1,
        backend="mock",
        quality_eval_method="random",
        output_dir="results/test"
    )

    runner = ExperimentRunner(config)

    # Use subset of example tasks
    tasks = EXAMPLE_TASKS[:2]

    # Run experiment
    results = runner.run_comparison_experiment(
        tasks=tasks,
        alpha_values=[0.0, 0.5, 1.0, 2.0]
    )

    # Analyze phase transition
    phase_analysis = runner.analyze_phase_transition(results)
    results["phase_transition"] = phase_analysis

    # Summarize
    summary = runner.summarize_results(results)
    results["summary"] = summary

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\nProtocol Comparison (Mean ± Std):")
    print("-" * 60)
    print(f"{'Protocol':<20} {'Diversity':<20} {'Quality':<20}")
    print("-" * 60)

    for protocol, stats in summary.get("protocol_comparison", {}).items():
        div_str = f"{stats['diversity_mean']:.3f} ± {stats['diversity_std']:.3f}"
        qual_str = f"{stats['quality_mean']:.3f} ± {stats['quality_std']:.3f}"
        print(f"{protocol:<20} {div_str:<20} {qual_str:<20}")

    print("-" * 60)

    if phase_analysis:
        print("\nPhase Transition Analysis:")
        cp = phase_analysis.get("critical_point", {})
        if cp.get("alpha_c"):
            print(f"  Critical point α_c: {cp['alpha_c']:.3f} (confidence: {cp['confidence']:.2f})")
        sl = phase_analysis.get("scaling_law", {})
        if sl.get("gamma"):
            print(f"  Scaling exponent γ: {sl['gamma']:.3f} (R²: {sl['r_squared']:.2f})")

    # Save results
    runner.save_results(results)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Density-Dependent Selection Experiments"
    )
    parser.add_argument(
        "--mode", choices=["test", "full"], default="test",
        help="Experiment mode: 'test' for quick verification, 'full' for complete run"
    )
    parser.add_argument(
        "--backend", choices=["mock", "openai", "anthropic"], default="mock",
        help="LLM backend to use"
    )
    parser.add_argument(
        "--n_agents", type=int, default=5,
        help="Number of agents"
    )
    parser.add_argument(
        "--n_rounds", type=int, default=3,
        help="Number of interaction rounds"
    )
    parser.add_argument(
        "--n_trials", type=int, default=3,
        help="Number of trials per condition"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    if args.mode == "test":
        run_quick_test()
    else:
        config = ExperimentConfig(
            name="full_experiment",
            n_agents=args.n_agents,
            n_rounds=args.n_rounds,
            n_trials=args.n_trials,
            backend=args.backend,
            output_dir=args.output_dir
        )

        runner = ExperimentRunner(config)
        results = runner.run_comparison_experiment(
            tasks=EXAMPLE_TASKS,
            alpha_values=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        )

        phase_analysis = runner.analyze_phase_transition(results)
        results["phase_transition"] = phase_analysis

        summary = runner.summarize_results(results)
        results["summary"] = summary

        runner.save_results(results)


if __name__ == "__main__":
    main()

"""
Extended Experiment Runner with MAP-Elites Comparison

Extends the base experiment runner to include:
1. MAP-Elites baseline comparison
2. Semantic Mapping for niche interpretation
3. Open-ended discovery metrics
4. Comprehensive comparison tables for paper

This supports the key paper claims:
- DDS discovers emergent niches (vs MAP-Elites pre-defined)
- Post-hoc axis interpretation (vs pre-defined behavior descriptors)
- Open-ended discovery (vs closed-ended niche filling)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from copy import deepcopy

from .llm_agent import (
    MultiAgentSystem, AgentConfig, Task, Response, EXAMPLE_TASKS
)
from .density_selection import (
    DensityDependentSelector, SelectionConfig, AgentResponse
)
from .map_elites import MAPElitesSelector, MAPElitesConfig, compare_map_elites_vs_dds
from .protocols import (
    DDSProtocol, DDSConfig, create_protocol, ProtocolConfig,
    NoveltySearchProtocol, NoveltySearchConfig
)
from .metrics import (
    compute_all_metrics, NicheMetrics, StatisticalTests,
    DownstreamEvaluator, compare_protocols_statistically
)
from .semantic_mapping import SemanticMapper, analyze_emergent_niches
from .openended_metrics import OpenEndedMetrics, OpenEndedExperiment


class ExtendedExperimentConfig:
    """Configuration for extended experiments with MAP-Elites comparison."""

    def __init__(
        self,
        name: str = "extended_experiment",
        n_agents: int = 5,
        n_rounds: int = 3,
        n_trials: int = 3,
        backend: str = "mock",
        model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        output_dir: str = "results",
        seed: int = 42,
        # DDS parameters
        dds_alpha_values: List[float] = None,
        dds_beta: float = 2.0,
        dds_bandwidth: float = 0.3,
        # MAP-Elites parameters
        map_n_bins: int = 5,
        map_behavior_dims: int = 2,
        map_use_cvt: bool = False,
        # Semantic mapping
        use_semantic_mapping: bool = True,
        semantic_n_features: int = 8
    ):
        self.name = name
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.n_trials = n_trials
        self.backend = backend
        self.model = model
        self.embedding_model = embedding_model
        self.output_dir = output_dir
        self.seed = seed

        self.dds_alpha_values = dds_alpha_values or [0.0, 0.5, 1.0, 2.0]
        self.dds_beta = dds_beta
        self.dds_bandwidth = dds_bandwidth

        self.map_n_bins = map_n_bins
        self.map_behavior_dims = map_behavior_dims
        self.map_use_cvt = map_use_cvt

        self.use_semantic_mapping = use_semantic_mapping
        self.semantic_n_features = semantic_n_features

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "n_agents": self.n_agents,
            "n_rounds": self.n_rounds,
            "n_trials": self.n_trials,
            "backend": self.backend,
            "model": self.model,
            "seed": self.seed,
            "dds_alpha_values": self.dds_alpha_values,
            "map_n_bins": self.map_n_bins
        }


class ExtendedExperimentRunner:
    """
    Extended experiment runner for DDS vs MAP-Elites comparison.

    Key experiments:
    1. Direct comparison on same responses
    2. Open-ended discovery evaluation
    3. Semantic axis interpretation
    """

    def __init__(self, config: ExtendedExperimentConfig):
        self.config = config
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

        # Initialize metrics
        self.openended_metrics = OpenEndedMetrics()

        # Setup output
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_direct_comparison(
        self,
        task: Task
    ) -> Dict:
        """
        Run direct comparison between DDS and MAP-Elites on same task.

        Returns detailed comparison metrics.
        """
        self.system.reset_all()

        # Generate initial responses
        responses = self.system.generate_responses(task)

        # Assign fixed quality score to neutralize quality term
        # This isolates the effect of density-dependent selection
        for r in responses:
            r.quality_score = 0.75

        # Convert to AgentResponse
        agent_responses = [
            AgentResponse(
                agent_id=r.agent_id,
                response_text=r.text,
                embedding=r.embedding,
                quality_score=r.quality_score
            )
            for r in responses
        ]

        # Run comparison
        dds_config = SelectionConfig(
            alpha=1.0,
            beta=self.config.dds_beta,
            bandwidth=self.config.dds_bandwidth
        )
        map_config = MAPElitesConfig(
            n_bins_per_dim=self.config.map_n_bins,
            n_behavior_dims=self.config.map_behavior_dims,
            use_cvt=self.config.map_use_cvt
        )

        comparison = compare_map_elites_vs_dds(
            agent_responses, dds_config, map_config
        )

        return comparison

    def run_multi_round_comparison(
        self,
        task: Task,
        n_rounds: int = None
    ) -> Dict:
        """
        Run multi-round experiment comparing DDS and MAP-Elites dynamics.

        Tracks niche formation over time.
        """
        if n_rounds is None:
            n_rounds = self.config.n_rounds

        self.system.reset_all()

        # Initialize selectors
        # Use first alpha value from config (typically α=1.0 for balanced selection)
        alpha = self.config.dds_alpha_values[0] if self.config.dds_alpha_values else 1.0
        dds_config = SelectionConfig(
            alpha=alpha,
            beta=self.config.dds_beta,
            bandwidth=self.config.dds_bandwidth
        )
        dds_selector = DensityDependentSelector(dds_config)

        map_config = MAPElitesConfig(
            n_bins_per_dim=self.config.map_n_bins,
            n_behavior_dims=self.config.map_behavior_dims
        )
        map_selector = MAPElitesSelector(map_config, behavior_method="pca")

        # Track history
        dds_history = []
        map_history = []

        # Initial round
        responses = self.system.generate_responses(task)
        for r in responses:
            r.quality_score = 0.75  # Fixed quality to isolate diversity mechanism

        agent_responses = [
            AgentResponse(
                agent_id=r.agent_id,
                response_text=r.text,
                embedding=r.embedding,
                quality_score=r.quality_score
            )
            for r in responses
        ]

        for round_idx in range(n_rounds):
            # DDS metrics
            embeddings = np.array([r.embedding for r in agent_responses])
            niche_result = NicheMetrics.count_niches_hierarchical(embeddings)
            dds_diversity = dds_selector.evaluate_diversity(agent_responses)

            dds_history.append({
                "round": round_idx,
                "n_niches": niche_result["n_niches"],
                "diversity": dds_diversity["mean_pairwise_distance"]
            })

            # MAP-Elites metrics
            map_stats = map_selector.update_archive(agent_responses)
            map_history.append({
                "round": round_idx,
                "n_elites": map_stats["n_elites"],
                "coverage": map_stats["coverage"],
                "qd_score": map_stats["qd_score"]
            })

            # DDS selection for next round
            if round_idx < n_rounds - 1:
                # Select 70% to maintain population diversity
                n_survive = max(3, int(len(agent_responses) * 0.7))
                selected, _ = dds_selector.select(
                    agent_responses,
                    n_select=n_survive
                )

                # Generate new responses based on selection
                # (In real experiment, this would involve LLM generation)
                # For now, simulate with perturbations
                new_responses = []
                for i, sel in enumerate(selected):
                    # Create variation
                    new_emb = sel.embedding + 0.1 * np.random.randn(len(sel.embedding))
                    new_emb = new_emb / np.linalg.norm(new_emb)

                    new_responses.append(AgentResponse(
                        agent_id=i,
                        response_text=f"Response {i} round {round_idx + 1}",
                        embedding=new_emb,
                        quality_score=0.75  # Fixed quality to isolate diversity mechanism
                    ))

                agent_responses = new_responses

        return {
            "dds_history": dds_history,
            "map_history": map_history,
            "final_comparison": {
                "dds_niches": dds_history[-1]["n_niches"],
                "map_elites": map_history[-1]["n_elites"],
                "dds_diversity": dds_history[-1]["diversity"],
                "map_coverage": map_history[-1]["coverage"]
            }
        }

    def run_alpha_sweep_comparison(
        self,
        task: Task,
        n_rounds: int = None,
        selection_ratio: float = 0.7,
        min_population: int = 5
    ) -> Dict:
        """
        Compare DDS at different alpha values vs MAP-Elites.

        Tests phase transition hypothesis by tracking metrics across rounds.

        Args:
            task: The task to run
            n_rounds: Number of selection rounds (default: self.config.n_rounds)
            selection_ratio: Fraction of population to keep each round (default: 0.7)
            min_population: Minimum population size to maintain (default: 5)

        Returns:
            Dict containing:
            - alpha_sweep: Per-alpha results with round-by-round metrics
            - map_baseline: MAP-Elites baseline metrics
            - phase_transition: Detected phase transition info (if any)
        """
        if n_rounds is None:
            n_rounds = self.config.n_rounds

        results = {"alpha_sweep": {}, "map_baseline": None, "phase_transition": None}

        self.system.reset_all()
        responses = self.system.generate_responses(task)
        for r in responses:
            r.quality_score = 0.75  # Fixed quality to isolate diversity mechanism

        agent_responses = [
            AgentResponse(
                agent_id=r.agent_id,
                response_text=r.text,
                embedding=r.embedding,
                quality_score=r.quality_score
            )
            for r in responses
        ]

        initial_population = len(agent_responses)

        # MAP-Elites baseline (alpha doesn't apply)
        map_config = MAPElitesConfig(
            n_bins_per_dim=self.config.map_n_bins,
            n_behavior_dims=self.config.map_behavior_dims
        )
        map_selector = MAPElitesSelector(map_config)
        map_selected, _ = map_selector.select(agent_responses, n_select=len(agent_responses))
        map_stats = map_selector.archive.get_statistics()

        results["map_baseline"] = {
            "n_elites": map_stats["n_elites"],
            "coverage": map_stats["coverage"],
            "qd_score": map_stats["qd_score"]
        }

        # DDS at different alphas - track metrics per round
        for alpha in self.config.dds_alpha_values:
            dds_config = SelectionConfig(
                alpha=alpha,
                beta=self.config.dds_beta,
                bandwidth=self.config.dds_bandwidth
            )
            dds_selector = DensityDependentSelector(dds_config)

            # Multi-round selection with per-round metrics
            current_responses = deepcopy(agent_responses)
            round_history = []

            for round_idx in range(n_rounds):
                # Compute metrics BEFORE selection (on current population)
                current_embeddings = np.array([r.embedding for r in current_responses])
                niche_result = NicheMetrics.count_niches_hierarchical(current_embeddings)
                diversity = dds_selector.evaluate_diversity(current_responses)

                round_history.append({
                    "round": round_idx,
                    "n_population": len(current_responses),
                    "n_niches": niche_result["n_niches"],
                    "diversity": diversity["mean_pairwise_distance"],
                    "cluster_sizes": niche_result.get("cluster_sizes", [])
                })

                # Selection step - maintain sufficient population
                n_select = max(
                    min_population,
                    int(len(current_responses) * selection_ratio)
                )
                # Don't select more than we have
                n_select = min(n_select, len(current_responses))

                if n_select < len(current_responses):
                    selected, selection_info = dds_selector.select(
                        current_responses,
                        n_select=n_select
                    )
                    current_responses = selected

            # Final metrics after all rounds
            final_embeddings = np.array([r.embedding for r in current_responses])
            final_niche_result = NicheMetrics.count_niches_hierarchical(final_embeddings)
            final_diversity = dds_selector.evaluate_diversity(current_responses)

            results["alpha_sweep"][str(alpha)] = {
                "n_niches": final_niche_result["n_niches"],
                "diversity": final_diversity["mean_pairwise_distance"],
                "n_final": len(current_responses),
                "round_history": round_history,
                # Summary statistics across rounds
                "max_niches": max(r["n_niches"] for r in round_history),
                "mean_diversity": float(np.mean([r["diversity"] for r in round_history])),
                "diversity_trend": round_history[-1]["diversity"] - round_history[0]["diversity"] if len(round_history) > 1 else 0.0
            }

        # Detect phase transition
        results["phase_transition"] = self._detect_phase_transition(results["alpha_sweep"])

        return results

    def _detect_phase_transition(self, alpha_sweep: Dict) -> Dict:
        """
        Detect phase transition in alpha sweep results.

        Looks for critical alpha value where behavior changes significantly.
        Detects both:
        1. Niche-based transitions (cluster count changes)
        2. Diversity-based transitions (diversity collapse at high alpha)
        """
        alphas = sorted([float(a) for a in alpha_sweep.keys()])
        if len(alphas) < 3:
            return {"detected": False, "reason": "Insufficient alpha values"}

        niches = [alpha_sweep[str(a)]["n_niches"] for a in alphas]
        diversities = [alpha_sweep[str(a)]["diversity"] for a in alphas]
        max_niches_list = [alpha_sweep[str(a)]["max_niches"] for a in alphas]

        # 1. Check for niche-based transition
        niche_diffs = np.diff(niches)
        if np.max(niches) > np.min(niches):
            max_diff_idx = np.argmax(np.abs(niche_diffs))
            critical_alpha = alphas[max_diff_idx + 1]
            return {
                "detected": True,
                "critical_alpha": critical_alpha,
                "niche_range": [int(np.min(niches)), int(np.max(niches))],
                "diversity_range": [float(np.min(diversities)), float(np.max(diversities))],
                "transition_type": "niche_emergence" if niche_diffs[max_diff_idx] > 0 else "niche_collapse",
                "max_niches_by_alpha": {str(a): max_niches_list[i] for i, a in enumerate(alphas)},
                "diversities_by_alpha": {str(a): diversities[i] for i, a in enumerate(alphas)}
            }

        # 2. Check for diversity-based transition (diversity collapse)
        # This is the key insight: high alpha causes diversity to collapse
        diversity_range = np.max(diversities) - np.min(diversities)
        if diversity_range > 0.05:  # Significant diversity change threshold
            diversity_diffs = np.diff(diversities)
            # Find the steepest drop in diversity
            max_drop_idx = np.argmin(diversity_diffs)  # Most negative = steepest drop

            # Check if this is a meaningful drop (not just noise)
            if diversity_diffs[max_drop_idx] < -0.03:  # Threshold for significant drop
                critical_alpha = alphas[max_drop_idx + 1]

                # Determine transition type based on direction
                if diversities[0] > diversities[-1]:
                    transition_type = "diversity_collapse"
                    description = f"Diversity collapses from {diversities[0]:.3f} to {diversities[-1]:.3f} as α increases"
                else:
                    transition_type = "diversity_emergence"
                    description = f"Diversity emerges from {diversities[0]:.3f} to {diversities[-1]:.3f} as α increases"

                return {
                    "detected": True,
                    "critical_alpha": critical_alpha,
                    "niche_range": [int(np.min(niches)), int(np.max(niches))],
                    "diversity_range": [float(np.min(diversities)), float(np.max(diversities))],
                    "transition_type": transition_type,
                    "description": description,
                    "max_niches_by_alpha": {str(a): max_niches_list[i] for i, a in enumerate(alphas)},
                    "diversities_by_alpha": {str(a): diversities[i] for i, a in enumerate(alphas)},
                    "optimal_alpha_range": self._find_optimal_alpha_range(alphas, diversities)
                }

        # 3. Also check max_niches (peak niches across rounds)
        if np.max(max_niches_list) > np.min(max_niches_list):
            max_niche_diffs = np.diff(max_niches_list)
            max_diff_idx = np.argmax(np.abs(max_niche_diffs))
            critical_alpha = alphas[max_diff_idx + 1]
            return {
                "detected": True,
                "critical_alpha": critical_alpha,
                "niche_range": [int(np.min(niches)), int(np.max(niches))],
                "max_niche_range": [int(np.min(max_niches_list)), int(np.max(max_niches_list))],
                "diversity_range": [float(np.min(diversities)), float(np.max(diversities))],
                "transition_type": "transient_niche_formation",
                "max_niches_by_alpha": {str(a): max_niches_list[i] for i, a in enumerate(alphas)},
                "diversities_by_alpha": {str(a): diversities[i] for i, a in enumerate(alphas)}
            }

        return {
            "detected": False,
            "reason": "No significant niche or diversity variation across alpha values",
            "niche_values": niches,
            "max_niche_values": max_niches_list,
            "diversity_values": diversities
        }

    def _find_optimal_alpha_range(self, alphas: List[float], diversities: List[float]) -> Dict:
        """
        Find the optimal alpha range that maximizes diversity.

        Returns the range of alpha values where diversity is above 80% of maximum.
        """
        max_diversity = np.max(diversities)
        threshold = max_diversity * 0.8

        optimal_alphas = [a for a, d in zip(alphas, diversities) if d >= threshold]

        if optimal_alphas:
            return {
                "min": float(min(optimal_alphas)),
                "max": float(max(optimal_alphas)),
                "peak_alpha": float(alphas[np.argmax(diversities)]),
                "peak_diversity": float(max_diversity)
            }
        return {"min": None, "max": None, "peak_alpha": None, "peak_diversity": float(max_diversity)}

    def run_semantic_interpretation(
        self,
        task: Task,
        alpha: float = 1.0
    ) -> Dict:
        """
        Run DDS and interpret emergent niches using Semantic Mapping.

        This demonstrates the key advantage: post-hoc axis interpretation.
        """
        if not self.config.use_semantic_mapping:
            return {"error": "Semantic mapping disabled"}

        self.system.reset_all()
        responses = self.system.generate_responses(task)
        for r in responses:
            r.quality_score = 0.75  # Fixed quality to isolate diversity mechanism

        # Run DDS
        dds_config = SelectionConfig(alpha=alpha, beta=self.config.dds_beta, bandwidth=self.config.dds_bandwidth)
        dds_selector = DensityDependentSelector(dds_config)

        agent_responses = [
            AgentResponse(
                agent_id=r.agent_id,
                response_text=r.text,
                embedding=r.embedding,
                quality_score=r.quality_score
            )
            for r in responses
        ]

        # Get cluster labels
        embeddings = np.array([r.embedding for r in agent_responses])
        niche_result = NicheMetrics.count_niches_hierarchical(embeddings)
        cluster_labels = np.array(niche_result["labels"])

        # Semantic interpretation
        response_tuples = [(r.agent_id, r.response_text) for r in agent_responses]

        interpretation = analyze_emergent_niches(
            responses=response_tuples,
            cluster_labels=cluster_labels,
            backend=self.config.backend
        )

        return {
            "n_niches": niche_result["n_niches"],
            "semantic_interpretation": interpretation,
            "key_insight": "Axes discovered POST-HOC from emergent niches"
        }

    def run_novelty_search_comparison(
        self,
        task: Task,
        n_rounds: int = None
    ) -> Dict:
        """
        Compare DDS with Novelty Search baseline (Lehman & Stanley 2011).

        Novelty Search selects purely based on novelty (ignoring quality).
        This tests whether density penalty with quality is better than pure novelty.
        """
        if n_rounds is None:
            n_rounds = self.config.n_rounds

        results = {"dds": {}, "novelty_search": {}, "comparison": {}}

        # Run DDS
        dds_config = DDSConfig(
            n_rounds=n_rounds,
            n_agents=self.config.n_agents,
            alpha=1.0,
            beta=self.config.dds_beta,
            bandwidth=self.config.dds_bandwidth,
            n_survive=max(3, int(self.config.n_agents * 0.6))
        )
        dds_protocol = DDSProtocol(dds_config)
        self.system.reset_all()
        dds_result = dds_protocol.run(self.system, task)

        # Run Novelty Search
        ns_config = NoveltySearchConfig(
            n_rounds=n_rounds,
            n_agents=self.config.n_agents,
            k_nearest=15,
            archive_threshold=0.3,
            n_survive=max(3, int(self.config.n_agents * 0.6))
        )
        ns_protocol = NoveltySearchProtocol(ns_config)
        self.system.reset_all()
        ns_result = ns_protocol.run(self.system, task)

        # Compute metrics for both
        dds_embeddings = np.array([r.embedding for r in dds_result["final_responses"]])
        ns_embeddings = np.array([r.embedding for r in ns_result["final_responses"]])

        dds_diversity = NicheMetrics.count_niches_hierarchical(dds_embeddings)
        ns_diversity = NicheMetrics.count_niches_hierarchical(ns_embeddings)

        dds_selector = DensityDependentSelector(SelectionConfig(alpha=1.0))
        agent_responses_dds = [
            AgentResponse(r.agent_id, r.text, r.embedding, r.quality_score)
            for r in dds_result["final_responses"]
        ]
        agent_responses_ns = [
            AgentResponse(r.agent_id, r.text, r.embedding, r.quality_score)
            for r in ns_result["final_responses"]
        ]

        results["dds"] = {
            "n_niches": dds_diversity["n_niches"],
            "diversity": dds_selector.evaluate_diversity(agent_responses_dds)["mean_pairwise_distance"],
            "mean_quality": float(np.mean([r.quality_score for r in dds_result["final_responses"]]))
        }

        results["novelty_search"] = {
            "n_niches": ns_diversity["n_niches"],
            "diversity": dds_selector.evaluate_diversity(agent_responses_ns)["mean_pairwise_distance"],
            "mean_quality": float(np.mean([r.quality_score for r in ns_result["final_responses"]])),
            "final_archive_size": ns_result.get("final_archive_size", 0)
        }

        # Statistical comparison
        results["comparison"] = {
            "diversity_diff": results["dds"]["diversity"] - results["novelty_search"]["diversity"],
            "quality_diff": results["dds"]["mean_quality"] - results["novelty_search"]["mean_quality"],
            "dds_better_diversity": results["dds"]["diversity"] > results["novelty_search"]["diversity"],
            "dds_better_quality": results["dds"]["mean_quality"] > results["novelty_search"]["mean_quality"]
        }

        return results

    def run_statistical_analysis(
        self,
        tasks: List[Task] = None,
        n_trials: int = 10
    ) -> Dict:
        """
        Run experiments with statistical significance tests.

        Compares DDS vs other baselines with proper statistical testing.
        """
        if tasks is None:
            tasks = EXAMPLE_TASKS[:2]

        # Collect results across trials
        protocol_results = {
            "dds": [],
            "novelty_search": [],
            "debate": [],
            "independent": []
        }

        for trial in range(n_trials):
            np.random.seed(self.config.seed + trial)

            for task in tasks:
                # DDS
                self.system.reset_all()
                dds_config = DDSConfig(n_rounds=self.config.n_rounds, n_agents=self.config.n_agents, alpha=1.0)
                dds_protocol = DDSProtocol(dds_config)
                dds_result = dds_protocol.run(self.system, task)
                dds_embeddings = np.array([r.embedding for r in dds_result["final_responses"]])
                dds_diversity = compute_all_metrics(
                    dds_embeddings,
                    [r.text for r in dds_result["final_responses"]],
                    [r.quality_score for r in dds_result["final_responses"]]
                )
                protocol_results["dds"].append(dds_diversity)

                # Novelty Search
                self.system.reset_all()
                ns_config = NoveltySearchConfig(n_rounds=self.config.n_rounds, n_agents=self.config.n_agents)
                ns_protocol = NoveltySearchProtocol(ns_config)
                ns_result = ns_protocol.run(self.system, task)
                ns_embeddings = np.array([r.embedding for r in ns_result["final_responses"]])
                ns_diversity = compute_all_metrics(
                    ns_embeddings,
                    [r.text for r in ns_result["final_responses"]],
                    [r.quality_score for r in ns_result["final_responses"]]
                )
                protocol_results["novelty_search"].append(ns_diversity)

                # Debate baseline
                self.system.reset_all()
                from .protocols import DebateConfig, DebateProtocol
                debate_config = DebateConfig(n_rounds=self.config.n_rounds, n_agents=self.config.n_agents)
                debate_protocol = DebateProtocol(debate_config)
                debate_result = debate_protocol.run(self.system, task)
                debate_embeddings = np.array([r.embedding for r in debate_result["final_responses"]])
                debate_diversity = compute_all_metrics(
                    debate_embeddings,
                    [r.text for r in debate_result["final_responses"]],
                    [r.quality_score for r in debate_result["final_responses"]]
                )
                protocol_results["debate"].append(debate_diversity)

                # Independent baseline
                self.system.reset_all()
                from .protocols import IndependentProtocol
                ind_config = ProtocolConfig(n_rounds=1, n_agents=self.config.n_agents)
                ind_protocol = IndependentProtocol(ind_config)
                ind_result = ind_protocol.run(self.system, task)
                ind_embeddings = np.array([r.embedding for r in ind_result["final_responses"]])
                ind_diversity = compute_all_metrics(
                    ind_embeddings,
                    [r.text for r in ind_result["final_responses"]],
                    [r.quality_score for r in ind_result["final_responses"]]
                )
                protocol_results["independent"].append(ind_diversity)

        # Statistical comparisons
        statistical_results = compare_protocols_statistically(
            protocol_results,
            metric_key="diversity.mean_distance"
        )

        # Quality-diversity tradeoff analysis
        qd_analysis = {}
        for protocol_name, results in protocol_results.items():
            diversities = [r["diversity"]["mean_distance"] for r in results]
            qualities = [r.get("quality", {}).get("mean", 0.5) for r in results]
            qd_analysis[protocol_name] = DownstreamEvaluator.diversity_quality_correlation(
                diversities, qualities
            )

        return {
            "statistical_comparisons": statistical_results,
            "qd_tradeoff": qd_analysis,
            "n_trials": n_trials,
            "n_tasks": len(tasks)
        }

    def run_downstream_evaluation(
        self,
        tasks: List[Task] = None
    ) -> Dict:
        """
        Evaluate downstream task performance as a function of diversity.

        Tests whether more diverse populations produce better outcomes.
        """
        if tasks is None:
            tasks = EXAMPLE_TASKS[:4]

        results = {
            "best_of_n": [],
            "ensemble": [],
            "coverage_quality_tradeoff": []
        }

        alpha_values = [0.0, 0.5, 1.0, 2.0]

        for task in tasks:
            for alpha in alpha_values:
                self.system.reset_all()

                # Run DDS with this alpha
                dds_config = DDSConfig(
                    n_rounds=self.config.n_rounds,
                    n_agents=self.config.n_agents,
                    alpha=alpha,
                    n_survive=max(3, int(self.config.n_agents * 0.7))
                )
                dds_protocol = DDSProtocol(dds_config)
                result = dds_protocol.run(self.system, task)

                # Compute diversity
                embeddings = np.array([r.embedding for r in result["final_responses"]])
                diversity_metrics = NicheMetrics.count_niches_hierarchical(embeddings)

                dds_selector = DensityDependentSelector(SelectionConfig(alpha=alpha))
                agent_responses = [
                    AgentResponse(r.agent_id, r.text, r.embedding, r.quality_score)
                    for r in result["final_responses"]
                ]
                diversity_score = dds_selector.evaluate_diversity(agent_responses)["mean_pairwise_distance"]

                # Best-of-N analysis
                quality_scores = [r.quality_score for r in result["final_responses"]]
                best_of_n = DownstreamEvaluator.best_of_n_quality(
                    quality_scores,
                    [diversity_score] * len(quality_scores)
                )
                best_of_n["alpha"] = alpha
                best_of_n["task_id"] = task.task_id
                best_of_n["diversity"] = diversity_score
                results["best_of_n"].append(best_of_n)

                # Coverage-quality tradeoff
                coverage = diversity_score  # Using diversity as coverage proxy
                quality = np.mean(quality_scores)
                results["coverage_quality_tradeoff"].append({
                    "alpha": alpha,
                    "task_id": task.task_id,
                    "coverage": coverage,
                    "quality": quality,
                    "n_niches": diversity_metrics["n_niches"]
                })

        # Analyze correlations
        all_diversities = [r["diversity"] for r in results["best_of_n"]]
        all_best_qualities = [r["best_quality"] for r in results["best_of_n"]]

        diversity_quality_corr = DownstreamEvaluator.diversity_quality_correlation(
            all_diversities, all_best_qualities
        )

        results["diversity_quality_correlation"] = diversity_quality_corr

        # Pareto analysis
        all_coverages = [r["coverage"] for r in results["coverage_quality_tradeoff"]]
        all_qualities = [r["quality"] for r in results["coverage_quality_tradeoff"]]
        pareto = DownstreamEvaluator.coverage_quality_tradeoff(all_coverages, all_qualities)
        results["pareto_analysis"] = pareto

        return results

    def run_code_generation_experiment(
        self,
        n_trials: int = 3
    ) -> Dict:
        """
        Run experiments specifically on code generation tasks.

        Tests DDS effectiveness on structured output tasks.
        """
        # Get code generation tasks
        code_tasks = [t for t in EXAMPLE_TASKS if t.category == "code_generation"]

        if not code_tasks:
            return {"error": "No code generation tasks found"}

        results = {
            "tasks": [],
            "summary": {}
        }

        for task in code_tasks:
            task_results = {"task_id": task.task_id, "trials": []}

            for trial in range(n_trials):
                np.random.seed(self.config.seed + trial)
                self.system.reset_all()

                # Run DDS
                dds_config = DDSConfig(
                    n_rounds=self.config.n_rounds,
                    n_agents=self.config.n_agents,
                    alpha=1.0
                )
                dds_protocol = DDSProtocol(dds_config)
                result = dds_protocol.run(self.system, task)

                # Compute metrics
                embeddings = np.array([r.embedding for r in result["final_responses"]])
                metrics = compute_all_metrics(
                    embeddings,
                    [r.text for r in result["final_responses"]],
                    [r.quality_score for r in result["final_responses"]]
                )

                task_results["trials"].append({
                    "trial": trial,
                    "diversity": metrics["diversity"]["mean_distance"],
                    "n_niches": metrics["niche_hierarchical"]["n_niches"],
                    "distinct_2": metrics["distinct_2"],
                    "mean_quality": metrics.get("quality", {}).get("mean", 0.5)
                })

            # Compute task summary with bootstrap CI
            diversities = [t["diversity"] for t in task_results["trials"]]
            qualities = [t["mean_quality"] for t in task_results["trials"]]

            task_results["summary"] = {
                "diversity": StatisticalTests.bootstrap_ci(diversities),
                "quality": StatisticalTests.bootstrap_ci(qualities)
            }

            results["tasks"].append(task_results)

        # Overall summary
        all_diversities = []
        all_qualities = []
        for task_result in results["tasks"]:
            all_diversities.extend([t["diversity"] for t in task_result["trials"]])
            all_qualities.extend([t["mean_quality"] for t in task_result["trials"]])

        results["summary"] = {
            "diversity": StatisticalTests.bootstrap_ci(all_diversities),
            "quality": StatisticalTests.bootstrap_ci(all_qualities),
            "n_tasks": len(code_tasks),
            "n_trials": n_trials
        }

        return results

    def run_full_experiment(
        self,
        tasks: List[Task] = None
    ) -> Dict:
        """
        Run complete experiment suite for paper.

        Includes:
        1. Direct comparison (Table 1)
        2. Alpha sweep (Phase transition figure)
        3. Multi-round dynamics (Time series figure)
        4. Semantic interpretation (Qualitative analysis)
        5. Open-ended metrics (Key contribution)
        6. Novelty Search comparison (NEW)
        7. Statistical analysis (NEW)
        8. Downstream evaluation (NEW)
        9. Code generation experiments (NEW)
        """
        if tasks is None:
            tasks = EXAMPLE_TASKS[:2]

        all_results = {
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "experiments": {}
        }

        print("\n" + "=" * 60)
        print("Extended Experiment: DDS vs MAP-Elites Comparison")
        print("=" * 60)

        # Experiment 1: Direct comparison
        print("\n[1/4] Direct Comparison...")
        direct_results = []
        for task in tqdm(tasks, desc="Tasks"):
            for trial in range(self.config.n_trials):
                result = self.run_direct_comparison(task)
                result["task_id"] = task.task_id
                result["trial"] = trial
                direct_results.append(result)
        all_results["experiments"]["direct_comparison"] = direct_results

        # Experiment 2: Alpha sweep
        print("\n[2/4] Alpha Sweep (Phase Transition)...")
        alpha_results = []
        for task in tqdm(tasks, desc="Tasks"):
            result = self.run_alpha_sweep_comparison(task)
            result["task_id"] = task.task_id
            alpha_results.append(result)
        all_results["experiments"]["alpha_sweep"] = alpha_results

        # Experiment 3: Multi-round dynamics
        print("\n[3/4] Multi-Round Dynamics...")
        dynamics_results = []
        for task in tqdm(tasks, desc="Tasks"):
            result = self.run_multi_round_comparison(task)
            result["task_id"] = task.task_id
            dynamics_results.append(result)
        all_results["experiments"]["dynamics"] = dynamics_results

        # Experiment 4: Semantic interpretation
        print("\n[4/8] Semantic Interpretation...")
        semantic_results = []
        for task in tqdm(tasks[:1], desc="Tasks"):  # Only first task for demo
            result = self.run_semantic_interpretation(task)
            result["task_id"] = task.task_id
            semantic_results.append(result)
        all_results["experiments"]["semantic"] = semantic_results

        # Experiment 5: Novelty Search comparison
        print("\n[5/8] Novelty Search Comparison...")
        novelty_results = []
        for task in tqdm(tasks, desc="Tasks"):
            result = self.run_novelty_search_comparison(task)
            result["task_id"] = task.task_id
            novelty_results.append(result)
        all_results["experiments"]["novelty_search"] = novelty_results

        # Experiment 6: Statistical analysis (with multiple trials)
        print("\n[6/8] Statistical Analysis...")
        stat_results = self.run_statistical_analysis(tasks=tasks, n_trials=5)
        all_results["experiments"]["statistical"] = stat_results

        # Experiment 7: Downstream evaluation
        print("\n[7/8] Downstream Evaluation...")
        downstream_results = self.run_downstream_evaluation(tasks=tasks)
        all_results["experiments"]["downstream"] = downstream_results

        # Experiment 8: Code generation experiments
        print("\n[8/8] Code Generation Experiments...")
        code_results = self.run_code_generation_experiment(n_trials=3)
        all_results["experiments"]["code_generation"] = code_results

        # Generate summary
        all_results["summary"] = self._generate_summary(all_results)

        return all_results

    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics for paper."""
        summary = {
            "key_findings": [],
            "table_data": {}
        }

        # Direct comparison summary
        direct = results["experiments"].get("direct_comparison", [])
        if direct:
            dds_diversities = [r["dds"]["diversity"]["mean_pairwise_distance"]
                              for r in direct]
            map_diversities = [r["map_elites"]["diversity"]["mean_pairwise_distance"]
                              for r in direct]

            summary["table_data"]["direct_comparison"] = {
                "dds_diversity_mean": float(np.mean(dds_diversities)),
                "dds_diversity_std": float(np.std(dds_diversities)),
                "map_diversity_mean": float(np.mean(map_diversities)),
                "map_diversity_std": float(np.std(map_diversities))
            }

            if np.mean(dds_diversities) > np.mean(map_diversities):
                summary["key_findings"].append(
                    "DDS achieves higher diversity than MAP-Elites"
                )

        # Alpha sweep summary
        alpha = results["experiments"].get("alpha_sweep", [])
        if alpha:
            # Check for phase transition using new detection (supports diversity-based)
            for task_result in alpha:
                phase_transition = task_result.get("phase_transition", {})
                if phase_transition.get("detected"):
                    critical_alpha = phase_transition.get("critical_alpha", "N/A")
                    transition_type = phase_transition.get("transition_type", "unknown")

                    # Handle diversity-based transitions
                    if transition_type in ["diversity_collapse", "diversity_emergence"]:
                        diversity_range = phase_transition.get("diversity_range", [0, 0])
                        description = phase_transition.get("description", "")
                        optimal_range = phase_transition.get("optimal_alpha_range", {})

                        finding = f"Phase transition (α_c={critical_alpha}): {transition_type}"
                        if optimal_range.get("peak_alpha") is not None:
                            finding += f", optimal α≈{optimal_range['peak_alpha']:.2f} (diversity={optimal_range['peak_diversity']:.3f})"
                        summary["key_findings"].append(finding)

                        # Add detailed transition info
                        summary["phase_transition_details"] = {
                            "critical_alpha": critical_alpha,
                            "transition_type": transition_type,
                            "diversity_range": diversity_range,
                            "optimal_alpha_range": optimal_range,
                            "description": description
                        }
                    else:
                        # Handle niche-based transitions
                        niche_range = phase_transition.get("niche_range", [0, 0])
                        max_niche_range = phase_transition.get("max_niche_range", niche_range)
                        summary["key_findings"].append(
                            f"Phase transition at α={critical_alpha}: "
                            f"{transition_type}, niches {max_niche_range[0]}→{max_niche_range[1]}"
                        )
                    break

            # Add alpha sweep table data
            if alpha:
                sweep_data = {}
                for task_result in alpha:
                    sweep = task_result.get("alpha_sweep", {})
                    for alpha_val, metrics in sweep.items():
                        if alpha_val not in sweep_data:
                            sweep_data[alpha_val] = {"n_niches": [], "diversity": [], "max_niches": []}
                        sweep_data[alpha_val]["n_niches"].append(metrics.get("n_niches", 0))
                        sweep_data[alpha_val]["diversity"].append(metrics.get("diversity", 0))
                        sweep_data[alpha_val]["max_niches"].append(metrics.get("max_niches", metrics.get("n_niches", 0)))

                summary["table_data"]["alpha_sweep"] = {
                    alpha_val: {
                        "n_niches_mean": float(np.mean(data["n_niches"])),
                        "diversity_mean": float(np.mean(data["diversity"])),
                        "max_niches_mean": float(np.mean(data["max_niches"]))
                    }
                    for alpha_val, data in sweep_data.items()
                }

        # Semantic interpretation summary
        semantic = results["experiments"].get("semantic", [])
        if semantic and semantic[0].get("semantic_interpretation"):
            axes = semantic[0]["semantic_interpretation"].get("axes", [])
            if axes:
                summary["key_findings"].append(
                    f"Emergent axes: {axes[0].get('label', 'PC1')} vs {axes[1].get('label', 'PC2') if len(axes) > 1 else 'PC2'}"
                )

        # Novelty Search comparison summary
        novelty = results["experiments"].get("novelty_search", [])
        if novelty:
            dds_better_count = sum(1 for r in novelty if r.get("comparison", {}).get("dds_better_diversity", False))
            total = len(novelty)
            if dds_better_count > total / 2:
                summary["key_findings"].append(
                    f"DDS outperforms Novelty Search in {dds_better_count}/{total} tasks on diversity"
                )

        # Statistical analysis summary
        stats = results["experiments"].get("statistical", {})
        if stats:
            comparisons = stats.get("statistical_comparisons", {}).get("pairwise_comparisons", {})
            for comparison_name, comparison in comparisons.items():
                if comparison.get("significant_005") and "dds" in comparison_name.lower():
                    effect = comparison.get("effect_interpretation", "unknown")
                    summary["key_findings"].append(
                        f"Significant difference ({comparison_name}): p<0.05, effect size: {effect}"
                    )
                    break

            summary["table_data"]["statistical"] = stats.get("statistical_comparisons", {}).get("summary", {})

        # Downstream evaluation summary
        downstream = results["experiments"].get("downstream", {})
        if downstream:
            corr = downstream.get("diversity_quality_correlation", {})
            if corr.get("pearson_p", 1.0) < 0.05:
                r = corr.get("pearson_r", 0.0)
                summary["key_findings"].append(
                    f"Diversity-quality correlation: r={r:.3f} (p<0.05)"
                )

            pareto = downstream.get("pareto_analysis", {})
            if pareto.get("pareto_optimal_count", 0) > 0:
                summary["table_data"]["pareto"] = {
                    "n_pareto_optimal": pareto["pareto_optimal_count"],
                    "hypervolume": pareto.get("hypervolume", 0.0)
                }

        # Code generation summary
        code_gen = results["experiments"].get("code_generation", {})
        if code_gen and not code_gen.get("error"):
            code_summary = code_gen.get("summary", {})
            diversity_ci = code_summary.get("diversity", {})
            if diversity_ci:
                summary["key_findings"].append(
                    f"Code generation diversity: {diversity_ci.get('mean', 0):.3f} "
                    f"[{diversity_ci.get('ci_lower', 0):.3f}, {diversity_ci.get('ci_upper', 0):.3f}]"
                )
            summary["table_data"]["code_generation"] = code_summary

        return summary

    def save_results(self, results: Dict, filename: str = None) -> str:
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"extended_experiment_{timestamp}.json"

        filepath = self.output_dir / filename

        # Convert numpy to list
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)

        print(f"\nResults saved: {filepath}")
        return str(filepath)


def run_quick_extended_test():
    """Run quick test of extended experiments."""
    print("=" * 60)
    print("Quick Test: Extended DDS vs MAP-Elites Comparison")
    print("=" * 60)

    config = ExtendedExperimentConfig(
        name="quick_extended_test",
        n_agents=5,
        n_rounds=2,
        n_trials=1,
        backend="mock",
        output_dir="results/extended_test",
        dds_alpha_values=[0.0, 0.5, 1.0, 2.0]
    )

    runner = ExtendedExperimentRunner(config)
    results = runner.run_full_experiment(tasks=EXAMPLE_TASKS[:2])

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    summary = results.get("summary", {})

    print("\nKey Findings:")
    for finding in summary.get("key_findings", []):
        print(f"  • {finding}")

    table = summary.get("table_data", {}).get("direct_comparison", {})
    if table:
        print("\nDirect Comparison (Diversity):")
        print(f"  DDS: {table.get('dds_diversity_mean', 0):.4f} ± {table.get('dds_diversity_std', 0):.4f}")
        print(f"  MAP: {table.get('map_diversity_mean', 0):.4f} ± {table.get('map_diversity_std', 0):.4f}")

    runner.save_results(results)

    return results


if __name__ == "__main__":
    run_quick_extended_test()

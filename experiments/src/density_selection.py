"""
Density-Dependent Selection for LLM Multi-Agent Systems

Core implementation of the density-dependent selection mechanism
as described in the paper methodology.

Key equations:
- Local density: ρ_i = Σ_j K(||φ(r_i) - φ(r_j)|| / h)
- Fitness: F_i = Q(r_i, T) - α · ρ_i
- Selection probability: P(select r_i) = exp(β·F_i) / Σ_j exp(β·F_j)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist


@dataclass
class SelectionConfig:
    """Configuration for density-dependent selection."""
    alpha: float = 1.0          # Competitive pressure strength
    beta: float = 1.0           # Selection temperature (inverse)
    bandwidth: float = 0.5      # Kernel bandwidth h
    kernel: str = "gaussian"    # Kernel type: "gaussian", "epanechnikov", "uniform"


@dataclass
class AgentResponse:
    """Represents an agent's response with its embedding and quality."""
    agent_id: int
    response_text: str
    embedding: np.ndarray
    quality_score: float
    generation: int = 0

    # Computed during selection
    local_density: float = 0.0
    fitness: float = 0.0
    selection_prob: float = 0.0


class DensityDependentSelector:
    """
    Implements density-dependent selection for multi-agent systems.

    This class computes local density in embedding space and applies
    competitive pressure to promote niche formation.
    """

    def __init__(self, config: SelectionConfig):
        self.config = config
        self._kernel_func = self._get_kernel_function(config.kernel)

    def _get_kernel_function(self, kernel_name: str) -> Callable:
        """Get the kernel function for density estimation."""
        if kernel_name == "gaussian":
            return lambda d: np.exp(-0.5 * d**2)
        elif kernel_name == "epanechnikov":
            return lambda d: np.maximum(0, 1 - d**2) * 0.75
        elif kernel_name == "uniform":
            return lambda d: (d <= 1).astype(float)
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")

    def compute_pairwise_distances(
        self,
        responses: List[AgentResponse]
    ) -> np.ndarray:
        """Compute pairwise distances between response embeddings."""
        embeddings = np.array([r.embedding for r in responses])
        distances = cdist(embeddings, embeddings, metric='cosine')
        return distances

    def compute_local_density(
        self,
        responses: List[AgentResponse],
        distances: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute local density for each response.

        ρ_i = Σ_{j≠i} K(||φ(r_i) - φ(r_j)|| / h)
        """
        if distances is None:
            distances = self.compute_pairwise_distances(responses)

        n = len(responses)
        h = self.config.bandwidth

        # Normalize distances by bandwidth
        normalized_distances = distances / h

        # Apply kernel
        kernel_values = self._kernel_func(normalized_distances)

        # Sum over all other agents (exclude self)
        np.fill_diagonal(kernel_values, 0)
        densities = kernel_values.sum(axis=1)

        return densities

    def compute_fitness(
        self,
        responses: List[AgentResponse],
        densities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute fitness with density penalty.

        F_i = Q(r_i, T) - α · ρ_i
        """
        if densities is None:
            densities = self.compute_local_density(responses)

        quality_scores = np.array([r.quality_score for r in responses])
        alpha = self.config.alpha

        fitness = quality_scores - alpha * densities

        return fitness

    def compute_selection_probabilities(
        self,
        fitness: np.ndarray
    ) -> np.ndarray:
        """
        Compute selection probabilities using softmax.

        P(select r_i) = exp(β·F_i) / Σ_j exp(β·F_j)
        """
        beta = self.config.beta

        # Numerical stability: subtract max
        scaled_fitness = beta * fitness
        scaled_fitness = scaled_fitness - scaled_fitness.max()

        exp_fitness = np.exp(scaled_fitness)
        probabilities = exp_fitness / exp_fitness.sum()

        return probabilities

    def select(
        self,
        responses: List[AgentResponse],
        n_select: Optional[int] = None,
        return_indices: bool = False
    ) -> Tuple[List[AgentResponse], np.ndarray]:
        """
        Perform density-dependent selection.

        Args:
            responses: List of agent responses
            n_select: Number of responses to select (default: same as input)
            return_indices: Whether to return selection indices

        Returns:
            Selected responses and their selection probabilities
        """
        if n_select is None:
            n_select = len(responses)

        # Compute all metrics
        distances = self.compute_pairwise_distances(responses)
        densities = self.compute_local_density(responses, distances)
        fitness = self.compute_fitness(responses, densities)
        probabilities = self.compute_selection_probabilities(fitness)

        # Update response objects with computed values
        for i, r in enumerate(responses):
            r.local_density = densities[i]
            r.fitness = fitness[i]
            r.selection_prob = probabilities[i]

        # Select based on probabilities
        indices = np.random.choice(
            len(responses),
            size=n_select,
            replace=True,
            p=probabilities
        )

        selected = [responses[i] for i in indices]

        if return_indices:
            return selected, probabilities, indices
        return selected, probabilities

    def evaluate_diversity(
        self,
        responses: List[AgentResponse]
    ) -> dict:
        """
        Compute diversity metrics for a set of responses.

        Returns:
            dict with:
            - mean_pairwise_distance: Average semantic distance
            - min_pairwise_distance: Minimum distance (closest pair)
            - max_pairwise_distance: Maximum distance (furthest pair)
            - std_pairwise_distance: Standard deviation
        """
        if len(responses) < 2:
            return {
                "mean_pairwise_distance": 0.0,
                "min_pairwise_distance": 0.0,
                "max_pairwise_distance": 0.0,
                "std_pairwise_distance": 0.0,
                "n_responses": len(responses)
            }

        distances = self.compute_pairwise_distances(responses)

        # Get upper triangle (exclude diagonal and duplicates)
        upper_tri = distances[np.triu_indices_from(distances, k=1)]

        if len(upper_tri) == 0:
            return {
                "mean_pairwise_distance": 0.0,
                "min_pairwise_distance": 0.0,
                "max_pairwise_distance": 0.0,
                "std_pairwise_distance": 0.0,
                "n_responses": len(responses)
            }

        return {
            "mean_pairwise_distance": float(upper_tri.mean()),
            "min_pairwise_distance": float(upper_tri.min()),
            "max_pairwise_distance": float(upper_tri.max()),
            "std_pairwise_distance": float(upper_tri.std()),
            "n_responses": len(responses)
        }


class FitnessSharingSelector(DensityDependentSelector):
    """
    Traditional Fitness Sharing baseline (Goldberg & Richardson, 1987).

    This is mathematically similar to density-dependent selection but
    uses multiplicative fitness sharing instead of additive penalty.

    Shared fitness: F'_i = F_i / Σ_j sh(d_ij)
    where sh(d) = 1 - (d/σ)^α if d < σ, else 0
    """

    def __init__(self, config: SelectionConfig, share_radius: float = 0.5):
        super().__init__(config)
        self.share_radius = share_radius

    def _sharing_function(self, distance: float) -> float:
        """Compute sharing function value."""
        if distance < self.share_radius:
            return 1 - (distance / self.share_radius) ** self.config.alpha
        return 0.0

    def compute_fitness(
        self,
        responses: List[AgentResponse],
        densities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute shared fitness (multiplicative sharing).

        F'_i = Q_i / m_i where m_i = Σ_j sh(d_ij)
        """
        distances = self.compute_pairwise_distances(responses)
        n = len(responses)

        # Compute niche counts
        niche_counts = np.zeros(n)
        for i in range(n):
            for j in range(n):
                niche_counts[i] += self._sharing_function(distances[i, j])

        # Avoid division by zero
        niche_counts = np.maximum(niche_counts, 1e-10)

        # Shared fitness
        quality_scores = np.array([r.quality_score for r in responses])
        shared_fitness = quality_scores / niche_counts

        return shared_fitness


def run_selection_round(
    responses: List[AgentResponse],
    selector: DensityDependentSelector,
    n_survive: int
) -> Tuple[List[AgentResponse], dict]:
    """
    Run one round of density-dependent selection.

    Returns:
        Tuple of (selected responses, metrics dict)
    """
    # Get diversity before selection
    pre_metrics = selector.evaluate_diversity(responses)

    # Perform selection
    selected, probs = selector.select(responses, n_select=n_survive)

    # Get diversity after selection
    post_metrics = selector.evaluate_diversity(selected)

    # Compile all metrics
    metrics = {
        "pre_selection": pre_metrics,
        "post_selection": post_metrics,
        "mean_fitness": float(np.mean([r.fitness for r in responses])),
        "mean_density": float(np.mean([r.local_density for r in responses])),
        "mean_quality": float(np.mean([r.quality_score for r in responses])),
        "selection_entropy": float(-np.sum(probs * np.log(probs + 1e-10)))
    }

    return selected, metrics


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)

    # Create synthetic responses with embeddings
    n_agents = 10
    embedding_dim = 64

    responses = []
    for i in range(n_agents):
        # Random embedding
        emb = np.random.randn(embedding_dim)
        emb = emb / np.linalg.norm(emb)  # Normalize

        # Random quality score
        quality = np.random.uniform(0.5, 1.0)

        responses.append(AgentResponse(
            agent_id=i,
            response_text=f"Response {i}",
            embedding=emb,
            quality_score=quality
        ))

    # Test density-dependent selection
    config = SelectionConfig(alpha=0.5, beta=2.0, bandwidth=0.3)
    selector = DensityDependentSelector(config)

    selected, metrics = run_selection_round(responses, selector, n_survive=5)

    print("Density-Dependent Selection Test")
    print("=" * 40)
    print(f"Input: {len(responses)} responses")
    print(f"Output: {len(selected)} selected")
    print(f"Pre-selection diversity: {metrics['pre_selection']['mean_pairwise_distance']:.4f}")
    print(f"Post-selection diversity: {metrics['post_selection']['mean_pairwise_distance']:.4f}")
    print(f"Mean fitness: {metrics['mean_fitness']:.4f}")
    print(f"Mean density: {metrics['mean_density']:.4f}")

"""
Open-Ended Discovery Metrics

Metrics to evaluate whether a method discovers "unexpected" or "novel" niches
that were not pre-defined. This is the key differentiator between:

- MAP-Elites: Fills pre-defined niches (closed-ended)
- DDS: Discovers emergent niches (open-ended)

Key metrics:
1. Niche Novelty Score: How different are discovered niches from expected ones?
2. Axis Surprise: How much do emergent axes differ from pre-defined ones?
3. Discovery Rate: Rate of finding new niches over time
4. Exploration Frontier: Coverage of unexplored semantic space
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .metrics import compute_pairwise_distances


class OpenEndedMetrics:
    """
    Metrics for evaluating open-ended discovery capabilities.
    """

    @staticmethod
    def niche_novelty_score(
        discovered_centroids: np.ndarray,  # (n_discovered, dim)
        expected_centroids: Optional[np.ndarray] = None,  # (n_expected, dim)
        reference_embeddings: Optional[np.ndarray] = None  # For generating expected
    ) -> Dict[str, float]:
        """
        Measure how novel the discovered niches are compared to expected ones.

        If no expected centroids provided, generate them using simple clustering
        on initial responses (simulating what MAP-Elites might pre-define).

        Higher score = more novel/unexpected discoveries
        """
        if discovered_centroids.shape[0] == 0:
            return {"novelty_score": 0.0, "mean_min_distance": 0.0}

        if expected_centroids is None:
            if reference_embeddings is None:
                # No reference: compare to uniform grid
                dim = discovered_centroids.shape[1]
                n_bins = 5
                grid_points = np.linspace(-1, 1, n_bins)
                expected_centroids = np.array(np.meshgrid(
                    *[grid_points[:2]] * min(dim, 2)
                )).T.reshape(-1, min(dim, 2))

                # Project discovered to same dimensions
                discovered_2d = discovered_centroids[:, :2]
            else:
                # Generate expected from reference using k-means
                n_expected = min(5, len(reference_embeddings))
                kmeans = KMeans(n_clusters=n_expected, random_state=42)
                kmeans.fit(reference_embeddings)
                expected_centroids = kmeans.cluster_centers_
                discovered_2d = discovered_centroids
        else:
            discovered_2d = discovered_centroids

        # Compute distances from discovered to nearest expected
        if expected_centroids.shape[1] != discovered_2d.shape[1]:
            # Dimension mismatch: project to common space
            min_dim = min(expected_centroids.shape[1], discovered_2d.shape[1])
            expected_centroids = expected_centroids[:, :min_dim]
            discovered_2d = discovered_2d[:, :min_dim]

        distances = cdist(discovered_2d, expected_centroids, metric='euclidean')
        min_distances = distances.min(axis=1)

        # Novelty score: average minimum distance (normalized)
        mean_min_distance = float(np.mean(min_distances))
        max_possible = np.sqrt(discovered_2d.shape[1]) * 2  # Rough upper bound
        novelty_score = mean_min_distance / max_possible

        return {
            "novelty_score": float(novelty_score),
            "mean_min_distance": mean_min_distance,
            "max_min_distance": float(np.max(min_distances)),
            "n_discovered": int(len(discovered_centroids)),
            "n_expected": int(len(expected_centroids))
        }

    @staticmethod
    def axis_surprise(
        emergent_pca: PCA,
        predefined_axes: np.ndarray  # (n_axes, embedding_dim)
    ) -> Dict[str, float]:
        """
        Measure how different emergent axes are from pre-defined ones.

        Uses cosine similarity between principal components and pre-defined axes.
        Lower similarity = higher surprise (more novel discovery)
        """
        emergent_axes = emergent_pca.components_  # (n_components, embedding_dim)

        # Handle dimension mismatch
        min_dim = min(emergent_axes.shape[1], predefined_axes.shape[1])
        emergent_axes = emergent_axes[:, :min_dim]
        predefined_axes = predefined_axes[:, :min_dim]

        # Normalize axes
        emergent_norm = emergent_axes / (np.linalg.norm(emergent_axes, axis=1, keepdims=True) + 1e-10)
        predefined_norm = predefined_axes / (np.linalg.norm(predefined_axes, axis=1, keepdims=True) + 1e-10)

        # Compute cosine similarities
        similarities = np.abs(emergent_norm @ predefined_norm.T)

        # Maximum similarity for each emergent axis
        max_similarities = similarities.max(axis=1)

        # Surprise = 1 - max_similarity (averaged)
        surprise = 1 - np.mean(max_similarities)

        return {
            "axis_surprise": float(surprise),
            "mean_max_similarity": float(np.mean(max_similarities)),
            "min_similarity": float(np.min(max_similarities)),
            "n_emergent_axes": int(len(emergent_axes)),
            "n_predefined_axes": int(len(predefined_axes))
        }

    @staticmethod
    def discovery_rate(
        niche_counts_over_time: List[int]
    ) -> Dict[str, float]:
        """
        Measure the rate of new niche discovery over iterations.

        Higher derivative = faster discovery
        Sustained discovery = open-ended behavior
        """
        if len(niche_counts_over_time) < 2:
            return {"discovery_rate": 0.0, "sustained_discovery": False}

        counts = np.array(niche_counts_over_time)

        # Compute derivative (new niches per round)
        derivatives = np.diff(counts)

        # Average discovery rate
        avg_rate = float(np.mean(derivatives))

        # Check if discovery is sustained (still finding new niches in later rounds)
        n_rounds = len(counts)
        late_rounds_start = max(1, n_rounds // 2)
        late_discovery = np.mean(derivatives[late_rounds_start:]) if late_rounds_start < len(derivatives) else 0

        sustained = late_discovery > 0

        return {
            "discovery_rate": avg_rate,
            "max_rate": float(np.max(derivatives)) if len(derivatives) > 0 else 0,
            "late_discovery_rate": float(late_discovery),
            "sustained_discovery": bool(sustained),
            "total_discovered": int(counts[-1] - counts[0]) if len(counts) > 0 else 0
        }

    @staticmethod
    def exploration_frontier(
        embeddings: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Measure coverage of the semantic space frontier.

        Uses PCA to project to 2D, then measures:
        - Grid coverage
        - Convex hull area
        - Spread from centroid
        """
        if len(embeddings) < 3:
            return {"frontier_coverage": 0.0, "spread": 0.0}

        # Project to 2D
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(embeddings)

        # Normalize to [-1, 1]
        min_vals = coords_2d.min(axis=0)
        max_vals = coords_2d.max(axis=0)
        range_vals = max_vals - min_vals + 1e-10
        normalized = 2 * (coords_2d - min_vals) / range_vals - 1

        # Grid coverage
        bins = np.linspace(-1, 1, n_bins + 1)
        hist, _, _ = np.histogram2d(normalized[:, 0], normalized[:, 1], bins=bins)
        occupied = np.sum(hist > 0)
        total_cells = n_bins * n_bins
        coverage = occupied / total_cells

        # Spread from centroid
        centroid = normalized.mean(axis=0)
        distances_from_centroid = np.linalg.norm(normalized - centroid, axis=1)
        spread = float(np.mean(distances_from_centroid))

        # Convex hull area (if scipy available)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(normalized)
            hull_area = hull.volume  # In 2D, volume is area
            max_area = 4.0  # Area of [-1,1] x [-1,1]
            relative_hull = hull_area / max_area
        except Exception:
            relative_hull = 0.0

        return {
            "frontier_coverage": float(coverage),
            "spread": spread,
            "hull_area": float(relative_hull),
            "explained_variance": float(sum(pca.explained_variance_ratio_)),
            "n_points": int(len(embeddings))
        }

    @staticmethod
    def niche_quality(
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        quality_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate quality of discovered niches.

        Good niches should:
        - Be well-separated (high silhouette)
        - Contain high-quality solutions
        - Be internally coherent
        """
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            return {
                "silhouette": 0.0,
                "mean_niche_quality": float(np.mean(quality_scores)),
                "quality_variance": float(np.var(quality_scores))
            }

        # Silhouette score
        try:
            sil_score = silhouette_score(embeddings, cluster_labels, metric='cosine')
        except Exception:
            sil_score = 0.0

        # Quality per niche
        niche_qualities = []
        for label in unique_labels:
            mask = cluster_labels == label
            if mask.sum() > 0:
                niche_qualities.append(np.mean(quality_scores[mask]))

        return {
            "silhouette": float(sil_score),
            "mean_niche_quality": float(np.mean(niche_qualities)),
            "best_niche_quality": float(np.max(niche_qualities)),
            "worst_niche_quality": float(np.min(niche_qualities)),
            "quality_variance_across_niches": float(np.var(niche_qualities)),
            "n_niches": n_clusters
        }


def compute_openended_comparison(
    dds_results: Dict,
    map_results: Dict,
    embeddings: np.ndarray
) -> Dict:
    """
    Compare DDS and MAP-Elites on open-ended discovery metrics.

    Arguments:
        dds_results: Results from DDS including cluster labels
        map_results: Results from MAP-Elites including archive
        embeddings: All response embeddings

    Returns:
        Comparison metrics highlighting open-ended capabilities
    """
    metrics = OpenEndedMetrics()

    # Get DDS niches
    dds_labels = np.array(dds_results.get("cluster_labels", []))
    if len(dds_labels) == 0:
        dds_labels = np.zeros(len(embeddings), dtype=int)

    # Compute DDS niche centroids
    dds_centroids = []
    for label in np.unique(dds_labels):
        mask = dds_labels == label
        if mask.sum() > 0:
            dds_centroids.append(embeddings[mask].mean(axis=0))
    dds_centroids = np.array(dds_centroids) if dds_centroids else np.array([]).reshape(0, embeddings.shape[1])

    # Get MAP-Elites archive centroids (pre-defined)
    map_centroids = np.array(map_results.get("behavior_centroids", []))
    if len(map_centroids) == 0:
        # Use grid-based centroids
        map_centroids = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5], [0, 0]])

    # Compute metrics
    dds_novelty = metrics.niche_novelty_score(
        dds_centroids[:, :2] if len(dds_centroids) > 0 else np.array([]).reshape(0, 2),
        map_centroids[:, :2] if len(map_centroids) > 0 else None,
        embeddings
    )

    dds_frontier = metrics.exploration_frontier(embeddings[dds_labels >= 0])

    # Compare discovery dynamics if available
    dds_niche_history = dds_results.get("niche_counts_history", [1])
    map_niche_history = map_results.get("archive_size_history", [1])

    dds_rate = metrics.discovery_rate(dds_niche_history)
    map_rate = metrics.discovery_rate(map_niche_history)

    return {
        "dds": {
            "novelty": dds_novelty,
            "frontier": dds_frontier,
            "discovery_rate": dds_rate
        },
        "map_elites": {
            "discovery_rate": map_rate
        },
        "comparison": {
            "dds_more_novel": dds_novelty["novelty_score"] > 0.1,
            "dds_higher_coverage": dds_frontier["frontier_coverage"] > 0.3,
            "dds_sustained_discovery": dds_rate["sustained_discovery"]
        }
    }


class OpenEndedExperiment:
    """
    Experiment framework for evaluating open-ended discovery.

    Compares DDS vs MAP-Elites on:
    1. Pre-defined niche filling (MAP-Elites advantage)
    2. Novel niche discovery (DDS advantage)
    3. Semantic interpretation (DDS + Semantic Mapping)
    """

    def __init__(self):
        self.metrics = OpenEndedMetrics()
        self.results_history = []

    def run_comparison(
        self,
        responses_with_embeddings: List[Tuple[int, str, np.ndarray, float]],
        n_rounds: int = 5,
        dds_alpha: float = 1.0
    ) -> Dict:
        """
        Run full open-ended comparison experiment.

        Args:
            responses_with_embeddings: List of (id, text, embedding, quality)
            n_rounds: Number of selection rounds
            dds_alpha: Competitive pressure for DDS

        Returns:
            Complete comparison results
        """
        from .density_selection import (
            DensityDependentSelector, SelectionConfig, AgentResponse
        )
        from .map_elites import MAPElitesSelector, MAPElitesConfig

        # Convert to AgentResponse objects
        agent_responses = [
            AgentResponse(
                agent_id=id,
                response_text=text,
                embedding=emb,
                quality_score=quality
            )
            for id, text, emb, quality in responses_with_embeddings
        ]

        embeddings = np.array([r.embedding for r in agent_responses])

        # Initialize selectors
        dds_config = SelectionConfig(alpha=dds_alpha, beta=2.0, bandwidth=0.3)
        dds_selector = DensityDependentSelector(dds_config)

        map_config = MAPElitesConfig(n_bins_per_dim=5, n_behavior_dims=2)
        map_selector = MAPElitesSelector(map_config, behavior_method="pca")

        # Track niche discovery over rounds
        dds_niche_history = []
        map_archive_history = []

        current_responses = agent_responses

        for round_idx in range(n_rounds):
            # DDS selection
            dds_selected, dds_probs = dds_selector.select(
                current_responses,
                n_select=len(current_responses) // 2
            )

            # Count DDS niches (using hierarchical clustering)
            from .metrics import NicheMetrics
            dds_embeddings = np.array([r.embedding for r in current_responses])
            niche_result = NicheMetrics.count_niches_hierarchical(dds_embeddings)
            dds_niche_history.append(niche_result["n_niches"])

            # MAP-Elites selection
            map_stats = map_selector.update_archive(current_responses)
            map_archive_history.append(map_stats["n_elites"])

        # Final analysis
        final_embeddings = np.array([r.embedding for r in current_responses])
        quality_scores = np.array([r.quality_score for r in current_responses])

        # Get final DDS clusters
        final_niche = NicheMetrics.count_niches_hierarchical(final_embeddings)
        dds_labels = np.array(final_niche["labels"])

        # Compute open-ended metrics
        dds_novelty = self.metrics.niche_novelty_score(
            self._compute_centroids(final_embeddings, dds_labels),
            reference_embeddings=embeddings
        )

        dds_frontier = self.metrics.exploration_frontier(final_embeddings)

        dds_quality = self.metrics.niche_quality(
            final_embeddings, dds_labels, quality_scores
        )

        dds_rate = self.metrics.discovery_rate(dds_niche_history)
        map_rate = self.metrics.discovery_rate(map_archive_history)

        results = {
            "dds": {
                "novelty": dds_novelty,
                "frontier": dds_frontier,
                "niche_quality": dds_quality,
                "discovery_rate": dds_rate,
                "final_n_niches": int(final_niche["n_niches"]),
                "niche_history": dds_niche_history
            },
            "map_elites": {
                "archive_stats": map_selector.archive.get_statistics(),
                "discovery_rate": map_rate,
                "archive_history": map_archive_history
            },
            "comparison": {
                "dds_novelty_advantage": dds_novelty["novelty_score"],
                "dds_coverage_advantage": dds_frontier["frontier_coverage"],
                "dds_sustained_discovery": dds_rate["sustained_discovery"],
                "map_coverage": map_selector.archive.get_coverage(),
                "conclusion": self._generate_conclusion(dds_novelty, dds_frontier, dds_rate)
            }
        }

        self.results_history.append(results)
        return results

    def _compute_centroids(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """Compute centroids for each cluster."""
        centroids = []
        for label in np.unique(labels):
            mask = labels == label
            if mask.sum() > 0:
                centroids.append(embeddings[mask].mean(axis=0))
        return np.array(centroids) if centroids else np.array([]).reshape(0, embeddings.shape[1])

    def _generate_conclusion(
        self,
        novelty: Dict,
        frontier: Dict,
        rate: Dict
    ) -> str:
        """Generate a textual conclusion about open-ended discovery."""
        if novelty["novelty_score"] > 0.2 and rate["sustained_discovery"]:
            return "DDS demonstrates strong open-ended discovery with novel niches"
        elif novelty["novelty_score"] > 0.1:
            return "DDS shows moderate novelty in niche discovery"
        else:
            return "DDS niches similar to pre-defined expectations"


if __name__ == "__main__":
    print("Testing Open-Ended Metrics")
    print("=" * 50)

    np.random.seed(42)

    # Create synthetic embeddings with clusters
    n_samples = 50
    embedding_dim = 64

    # Generate clustered data
    embeddings = []
    for cluster in range(5):
        center = np.random.randn(embedding_dim)
        center = center / np.linalg.norm(center)
        for _ in range(n_samples // 5):
            point = center + 0.1 * np.random.randn(embedding_dim)
            point = point / np.linalg.norm(point)
            embeddings.append(point)

    embeddings = np.array(embeddings)
    cluster_labels = np.repeat(np.arange(5), n_samples // 5)
    quality_scores = np.random.uniform(0.5, 1.0, n_samples)

    metrics = OpenEndedMetrics()

    # Test novelty score
    centroids = []
    for i in range(5):
        mask = cluster_labels == i
        centroids.append(embeddings[mask].mean(axis=0))
    centroids = np.array(centroids)

    novelty = metrics.niche_novelty_score(centroids, reference_embeddings=embeddings)
    print(f"\nNovelty Score: {novelty['novelty_score']:.3f}")

    # Test frontier coverage
    frontier = metrics.exploration_frontier(embeddings)
    print(f"Frontier Coverage: {frontier['frontier_coverage']:.3f}")
    print(f"Spread: {frontier['spread']:.3f}")

    # Test niche quality
    quality = metrics.niche_quality(embeddings, cluster_labels, quality_scores)
    print(f"Silhouette Score: {quality['silhouette']:.3f}")
    print(f"Mean Niche Quality: {quality['mean_niche_quality']:.3f}")

    # Test discovery rate
    niche_history = [1, 2, 3, 4, 5, 5, 6, 6]
    rate = metrics.discovery_rate(niche_history)
    print(f"Discovery Rate: {rate['discovery_rate']:.3f}")
    print(f"Sustained Discovery: {rate['sustained_discovery']}")

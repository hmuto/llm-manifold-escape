"""
MAP-Elites Baseline Implementation

Implements MAP-Elites (Multi-dimensional Archive of Phenotypic Elites)
as a baseline for comparison with Density-Dependent Selection.

Key differences from DDS:
- MAP-Elites: Pre-defined behavior descriptors, passive archiving
- DDS: Emergent niches in embedding space, active repulsion

Reference: Mouret & Clune (2015) "Illuminating search spaces by mapping elites"
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from .density_selection import AgentResponse, SelectionConfig
from .metrics import compute_pairwise_distances


@dataclass
class MAPElitesConfig:
    """Configuration for MAP-Elites."""
    n_bins_per_dim: int = 5          # Number of bins per behavior dimension
    n_behavior_dims: int = 2          # Number of behavior descriptor dimensions
    behavior_bounds: Tuple[float, float] = (-1.0, 1.0)  # Bounds for each dimension
    use_cvt: bool = False             # Use CVT-MAP-Elites (centroidal Voronoi tessellation)
    n_cvt_centroids: int = 100        # Number of centroids for CVT


@dataclass
class ArchiveCell:
    """A cell in the MAP-Elites archive."""
    response: Optional[AgentResponse] = None
    fitness: float = float('-inf')
    behavior: Optional[np.ndarray] = None

    def is_empty(self) -> bool:
        return self.response is None


class BehaviorDescriptor:
    """
    Abstract behavior descriptor that maps responses to behavior space.

    In standard MAP-Elites, this is pre-defined by humans.
    We provide several options for comparison:
    1. PCA-based: First 2 PCA components of embedding (requires global fit)
    2. Pre-defined features: Human-specified semantic dimensions
    3. Random projection: For baseline comparison
    """

    def __init__(self, method: str = "pca", n_dims: int = 2):
        self.method = method
        self.n_dims = n_dims
        self._pca = None
        self._projection_matrix = None
        self._fitted = False

    def fit(self, embeddings: np.ndarray):
        """Fit the behavior descriptor to a set of embeddings."""
        if self.method == "pca":
            from sklearn.decomposition import PCA
            self._pca = PCA(n_components=self.n_dims)
            self._pca.fit(embeddings)
        elif self.method == "random":
            # Random projection matrix
            np.random.seed(42)  # For reproducibility
            self._projection_matrix = np.random.randn(embeddings.shape[1], self.n_dims)
            self._projection_matrix /= np.linalg.norm(self._projection_matrix, axis=0)
        self._fitted = True

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to behavior descriptors."""
        if not self._fitted:
            self.fit(embeddings)

        if self.method == "pca":
            return self._pca.transform(embeddings)
        elif self.method == "random":
            return embeddings @ self._projection_matrix
        else:
            # Default: use first n_dims of embedding
            return embeddings[:, :self.n_dims]

    def get_explained_variance(self) -> Optional[np.ndarray]:
        """Get explained variance ratio (PCA only)."""
        if self.method == "pca" and self._pca is not None:
            return self._pca.explained_variance_ratio_
        return None


class MAPElitesArchive:
    """
    The MAP-Elites archive that stores elite solutions.

    Standard version uses a grid-based archive.
    CVT version uses centroidal Voronoi tessellation for better coverage.
    """

    def __init__(self, config: MAPElitesConfig):
        self.config = config

        if config.use_cvt:
            self._init_cvt_archive()
        else:
            self._init_grid_archive()

    def _init_grid_archive(self):
        """Initialize grid-based archive."""
        self.archive_type = "grid"
        shape = tuple([self.config.n_bins_per_dim] * self.config.n_behavior_dims)
        self.cells = {}  # Dict mapping cell index to ArchiveCell
        self.shape = shape

    def _init_cvt_archive(self):
        """Initialize CVT-based archive."""
        self.archive_type = "cvt"
        # Generate centroids using k-means on uniform samples
        n_samples = self.config.n_cvt_centroids * 100
        bounds = self.config.behavior_bounds
        samples = np.random.uniform(
            bounds[0], bounds[1],
            (n_samples, self.config.n_behavior_dims)
        )
        kmeans = KMeans(n_clusters=self.config.n_cvt_centroids, random_state=42)
        kmeans.fit(samples)
        self.centroids = kmeans.cluster_centers_
        self.cells = {i: ArchiveCell() for i in range(self.config.n_cvt_centroids)}

    def _behavior_to_cell_index(self, behavior: np.ndarray) -> tuple:
        """Convert behavior descriptor to cell index."""
        if self.archive_type == "grid":
            bounds = self.config.behavior_bounds
            n_bins = self.config.n_bins_per_dim

            # Normalize to [0, 1]
            normalized = (behavior - bounds[0]) / (bounds[1] - bounds[0])
            normalized = np.clip(normalized, 0, 0.999)  # Avoid edge case

            # Convert to bin indices
            indices = (normalized * n_bins).astype(int)
            return tuple(indices)
        else:
            # CVT: find nearest centroid
            distances = cdist([behavior], self.centroids)[0]
            return int(np.argmin(distances))

    def add(self, response: AgentResponse, behavior: np.ndarray, fitness: float) -> bool:
        """
        Try to add a response to the archive.

        Returns True if the response was added (new cell or better fitness).
        """
        cell_idx = self._behavior_to_cell_index(behavior)

        if cell_idx not in self.cells:
            self.cells[cell_idx] = ArchiveCell()

        cell = self.cells[cell_idx]

        # Add if cell is empty or new solution is better
        if cell.is_empty() or fitness > cell.fitness:
            self.cells[cell_idx] = ArchiveCell(
                response=response,
                fitness=fitness,
                behavior=behavior
            )
            return True
        return False

    def get_all_elites(self) -> List[AgentResponse]:
        """Get all elite responses from the archive."""
        return [cell.response for cell in self.cells.values()
                if not cell.is_empty()]

    def get_coverage(self) -> float:
        """Get proportion of cells that are filled."""
        n_filled = sum(1 for cell in self.cells.values() if not cell.is_empty())
        if self.archive_type == "grid":
            total_cells = np.prod(self.shape)
        else:
            total_cells = self.config.n_cvt_centroids
        return n_filled / total_cells

    def get_qd_score(self) -> float:
        """
        Get Quality-Diversity score (sum of all elite fitnesses).
        This is the standard MAP-Elites performance metric.
        """
        return sum(cell.fitness for cell in self.cells.values()
                   if not cell.is_empty())

    def get_statistics(self) -> Dict:
        """Get archive statistics."""
        elites = self.get_all_elites()
        fitnesses = [cell.fitness for cell in self.cells.values()
                     if not cell.is_empty()]

        return {
            "n_elites": len(elites),
            "coverage": self.get_coverage(),
            "qd_score": self.get_qd_score(),
            "mean_fitness": np.mean(fitnesses) if fitnesses else 0.0,
            "max_fitness": np.max(fitnesses) if fitnesses else 0.0,
            "min_fitness": np.min(fitnesses) if fitnesses else 0.0
        }


class MAPElitesSelector:
    """
    MAP-Elites selection mechanism for multi-agent systems.

    Key limitation (for paper): Requires pre-defined behavior descriptors.
    """

    def __init__(
        self,
        config: MAPElitesConfig,
        behavior_method: str = "pca"
    ):
        self.config = config
        self.archive = MAPElitesArchive(config)
        self.behavior_descriptor = BehaviorDescriptor(
            method=behavior_method,
            n_dims=config.n_behavior_dims
        )
        self._initial_fit_done = False

    def _ensure_behavior_fitted(self, responses: List[AgentResponse]):
        """Ensure behavior descriptor is fitted."""
        if not self._initial_fit_done:
            embeddings = np.array([r.embedding for r in responses])
            self.behavior_descriptor.fit(embeddings)
            self._initial_fit_done = True

    def update_archive(self, responses: List[AgentResponse]) -> Dict:
        """
        Update the archive with new responses.

        Returns statistics about the update.
        """
        self._ensure_behavior_fitted(responses)

        embeddings = np.array([r.embedding for r in responses])
        behaviors = self.behavior_descriptor.transform(embeddings)

        n_added = 0
        for i, response in enumerate(responses):
            added = self.archive.add(
                response=response,
                behavior=behaviors[i],
                fitness=response.quality_score
            )
            if added:
                n_added += 1

        return {
            "n_added": n_added,
            "n_total": len(responses),
            **self.archive.get_statistics()
        }

    def select(
        self,
        responses: List[AgentResponse],
        n_select: int
    ) -> Tuple[List[AgentResponse], np.ndarray]:
        """
        Select responses using MAP-Elites logic.

        Selection is based on:
        1. Update archive with all responses
        2. Select from archive (elites) with uniform probability

        This differs from DDS which uses density-based selection.
        """
        # Update archive
        self.update_archive(responses)

        # Get all elites
        elites = self.archive.get_all_elites()

        if len(elites) == 0:
            # Fallback: return random selection
            indices = np.random.choice(len(responses), size=n_select, replace=True)
            probs = np.ones(len(responses)) / len(responses)
            return [responses[i] for i in indices], probs

        # Select uniformly from elites (standard MAP-Elites behavior)
        # This encourages exploration of all niches equally
        probs = np.ones(len(elites)) / len(elites)
        indices = np.random.choice(len(elites), size=n_select, replace=True)

        return [elites[i] for i in indices], probs

    def get_niche_analysis(self) -> Dict:
        """
        Analyze the niches discovered by MAP-Elites.

        For comparison with DDS emergent niches.
        """
        elites = self.archive.get_all_elites()
        if not elites:
            return {"n_niches": 0, "behaviors": []}

        behaviors = []
        for cell in self.archive.cells.values():
            if not cell.is_empty():
                behaviors.append(cell.behavior.tolist())

        return {
            "n_niches": len(elites),
            "behaviors": behaviors,
            "explained_variance": self.behavior_descriptor.get_explained_variance(),
            "archive_stats": self.archive.get_statistics()
        }


def compare_map_elites_vs_dds(
    responses: List[AgentResponse],
    dds_config: SelectionConfig,
    map_config: MAPElitesConfig
) -> Dict:
    """
    Direct comparison between MAP-Elites and DDS on the same responses.

    Returns metrics comparing both approaches.
    """
    from .density_selection import DensityDependentSelector

    # DDS selection
    dds_selector = DensityDependentSelector(dds_config)
    dds_selected, dds_probs = dds_selector.select(responses, n_select=len(responses)//2)
    dds_diversity = dds_selector.evaluate_diversity(dds_selected)

    # MAP-Elites selection
    map_selector = MAPElitesSelector(map_config)
    map_selected, map_probs = map_selector.select(responses, n_select=len(responses)//2)

    # Compute diversity for MAP-Elites selected
    if map_selected:
        map_embeddings = np.array([r.embedding for r in map_selected])
        map_distances = compute_pairwise_distances(map_embeddings)
        upper_tri = map_distances[np.triu_indices_from(map_distances, k=1)]
        map_diversity = {
            "mean_pairwise_distance": float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0,
            "n_responses": len(map_selected)
        }
    else:
        map_diversity = {"mean_pairwise_distance": 0, "n_responses": 0}

    return {
        "dds": {
            "diversity": dds_diversity,
            "selection_entropy": float(-np.sum(dds_probs * np.log(dds_probs + 1e-10)))
        },
        "map_elites": {
            "diversity": map_diversity,
            "archive_stats": map_selector.archive.get_statistics(),
            "niche_analysis": map_selector.get_niche_analysis()
        }
    }


if __name__ == "__main__":
    # Test MAP-Elites implementation
    print("Testing MAP-Elites Implementation")
    print("=" * 50)

    np.random.seed(42)

    # Create synthetic responses
    n_responses = 50
    embedding_dim = 64

    responses = []
    for i in range(n_responses):
        # Create clustered embeddings to simulate niches
        cluster = i % 5
        center = np.zeros(embedding_dim)
        center[cluster * 10:(cluster + 1) * 10] = 1.0

        emb = center + 0.1 * np.random.randn(embedding_dim)
        emb = emb / np.linalg.norm(emb)

        quality = np.random.uniform(0.5, 1.0)

        responses.append(AgentResponse(
            agent_id=i,
            response_text=f"Response {i}",
            embedding=emb,
            quality_score=quality
        ))

    # Test grid-based MAP-Elites
    print("\nGrid-based MAP-Elites:")
    config = MAPElitesConfig(n_bins_per_dim=5, n_behavior_dims=2)
    selector = MAPElitesSelector(config, behavior_method="pca")

    selected, probs = selector.select(responses, n_select=10)
    stats = selector.archive.get_statistics()

    print(f"  Elites in archive: {stats['n_elites']}")
    print(f"  Coverage: {stats['coverage']:.2%}")
    print(f"  QD Score: {stats['qd_score']:.3f}")
    print(f"  Mean fitness: {stats['mean_fitness']:.3f}")

    # Test CVT-MAP-Elites
    print("\nCVT-MAP-Elites:")
    cvt_config = MAPElitesConfig(use_cvt=True, n_cvt_centroids=25, n_behavior_dims=2)
    cvt_selector = MAPElitesSelector(cvt_config, behavior_method="pca")

    cvt_selected, cvt_probs = cvt_selector.select(responses, n_select=10)
    cvt_stats = cvt_selector.archive.get_statistics()

    print(f"  Elites in archive: {cvt_stats['n_elites']}")
    print(f"  Coverage: {cvt_stats['coverage']:.2%}")
    print(f"  QD Score: {cvt_stats['qd_score']:.3f}")

    # Compare with DDS
    print("\nComparison with DDS:")
    from .density_selection import SelectionConfig
    dds_config = SelectionConfig(alpha=1.0, beta=2.0, bandwidth=0.3)
    comparison = compare_map_elites_vs_dds(responses, dds_config, config)

    print(f"  DDS diversity: {comparison['dds']['diversity']['mean_pairwise_distance']:.4f}")
    print(f"  MAP-Elites diversity: {comparison['map_elites']['diversity']['mean_pairwise_distance']:.4f}")

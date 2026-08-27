"""
Evaluation Metrics for Multi-Agent Experiments

Implements metrics for assessing:
1. Semantic Diversity: Pairwise distances in embedding space
2. Quality Score: Task-specific evaluation
3. Niche Count: Number of distinct clusters
4. Coverage: Proportion of solution space explored
5. Phase Transition Detection: Critical point analysis
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from collections import Counter
import warnings


def compute_pairwise_distances(
    embeddings: np.ndarray,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Compute pairwise distances between embeddings.

    Args:
        embeddings: (N, D) array of embeddings
        metric: Distance metric (cosine, euclidean, etc.)

    Returns:
        (N, N) distance matrix
    """
    if metric == "cosine":
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        normalized = embeddings / norms
        # Cosine distance = 1 - cosine_similarity
        similarity = normalized @ normalized.T
        distances = 1 - similarity
        # Ensure non-negative (numerical issues)
        distances = np.maximum(distances, 0)
    else:
        distances = squareform(pdist(embeddings, metric=metric))

    return distances


class DiversityMetrics:
    """
    Compute various diversity metrics for a set of responses.
    """

    @staticmethod
    def semantic_diversity(
        embeddings: np.ndarray,
        metric: str = "cosine"
    ) -> Dict[str, float]:
        """
        Compute semantic diversity metrics.

        Returns:
            Dictionary with:
            - mean_distance: Average pairwise distance
            - min_distance: Minimum pairwise distance
            - max_distance: Maximum pairwise distance
            - std_distance: Standard deviation of distances
        """
        distances = compute_pairwise_distances(embeddings, metric)
        n = len(embeddings)

        # Get upper triangle (exclude diagonal)
        upper_tri_indices = np.triu_indices(n, k=1)
        upper_tri = distances[upper_tri_indices]

        if len(upper_tri) == 0:
            return {
                "mean_distance": 0.0,
                "min_distance": 0.0,
                "max_distance": 0.0,
                "std_distance": 0.0
            }

        return {
            "mean_distance": float(np.mean(upper_tri)),
            "min_distance": float(np.min(upper_tri)),
            "max_distance": float(np.max(upper_tri)),
            "std_distance": float(np.std(upper_tri))
        }

    @staticmethod
    def distinct_n(
        texts: List[str],
        n: int = 2
    ) -> float:
        """
        Compute Distinct-N metric (ratio of unique n-grams).

        This metric was used in Paper 1 (EIDWT) for diversity measurement.
        """
        all_ngrams = []
        for text in texts:
            words = text.lower().split()
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            all_ngrams.extend(ngrams)

        if len(all_ngrams) == 0:
            return 0.0

        unique_ngrams = set(all_ngrams)
        return len(unique_ngrams) / len(all_ngrams)

    @staticmethod
    def jensen_shannon_divergence(
        embeddings: np.ndarray,
        n_bins: int = 50
    ) -> float:
        """
        Compute Jensen-Shannon Divergence from uniform distribution.

        Lower values indicate more uniform (diverse) distribution.
        This metric was used in Paper 2 for diversity measurement.
        """
        # Use PCA to reduce to 2D for binning
        from sklearn.decomposition import PCA

        if embeddings.shape[0] < 2:
            return 0.0

        n_components = min(2, embeddings.shape[1], embeddings.shape[0])
        pca = PCA(n_components=n_components)

        try:
            reduced = pca.fit_transform(embeddings)
        except Exception:
            return 0.0

        # Create 2D histogram
        if n_components == 2:
            hist, _, _ = np.histogram2d(
                reduced[:, 0], reduced[:, 1],
                bins=n_bins, density=True
            )
        else:
            hist, _ = np.histogram(reduced[:, 0], bins=n_bins, density=True)

        # Flatten and normalize
        p = hist.flatten()
        p = p / (p.sum() + 1e-10)

        # Uniform distribution
        q = np.ones_like(p) / len(p)

        # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
        m = 0.5 * (p + q)

        # Avoid log(0)
        p = np.maximum(p, 1e-10)
        q = np.maximum(q, 1e-10)
        m = np.maximum(m, 1e-10)

        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))

        jsd = 0.5 * (kl_pm + kl_qm)

        return float(jsd)


class NicheMetrics:
    """
    Compute niche-related metrics for assessing cluster formation.
    """

    @staticmethod
    def count_niches_dbscan(
        embeddings: np.ndarray,
        eps: float = 0.3,
        min_samples: int = 2,
        metric: str = "cosine"
    ) -> Dict[str, any]:
        """
        Count number of niches using DBSCAN clustering.

        Returns:
            Dictionary with:
            - n_niches: Number of clusters (excluding noise)
            - n_noise: Number of noise points
            - niche_sizes: List of cluster sizes
        """
        distances = compute_pairwise_distances(embeddings, metric)

        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="precomputed"
        ).fit(distances)

        labels = clustering.labels_

        # Count clusters (excluding noise, labeled as -1)
        unique_labels = set(labels)
        n_niches = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels == -1)

        # Get cluster sizes
        niche_sizes = []
        for label in unique_labels:
            if label != -1:
                niche_sizes.append(int(np.sum(labels == label)))

        return {
            "n_niches": n_niches,
            "n_noise": int(n_noise),
            "niche_sizes": niche_sizes,
            "labels": labels.tolist()
        }

    @staticmethod
    def count_niches_hierarchical(
        embeddings: np.ndarray,
        threshold: float = 0.5,
        metric: str = "cosine",
        linkage_method: str = "average"
    ) -> Dict[str, any]:
        """
        Count number of niches using hierarchical clustering.
        """
        if len(embeddings) < 2:
            return {
                "n_niches": len(embeddings),
                "niche_sizes": [1] * len(embeddings),
                "labels": list(range(len(embeddings)))
            }

        # Compute condensed distance matrix
        distances = compute_pairwise_distances(embeddings, metric)
        # Ensure diagonal is zero (numerical precision)
        np.fill_diagonal(distances, 0)
        condensed = squareform(distances)

        # Hierarchical clustering
        Z = linkage(condensed, method=linkage_method)
        labels = fcluster(Z, t=threshold, criterion='distance')

        # Count clusters
        label_counts = Counter(labels)
        n_niches = len(label_counts)
        niche_sizes = list(label_counts.values())

        return {
            "n_niches": n_niches,
            "niche_sizes": niche_sizes,
            "labels": labels.tolist()
        }

    @staticmethod
    def silhouette_score(
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute silhouette score for clustering quality.

        Higher values indicate better-defined clusters.
        """
        from sklearn.metrics import silhouette_score as sklearn_silhouette

        unique_labels = set(labels)
        if len(unique_labels) < 2:
            return 0.0

        try:
            score = sklearn_silhouette(embeddings, labels, metric='cosine')
            return float(score)
        except Exception:
            return 0.0


class CoverageMetrics:
    """
    Compute coverage metrics for assessing solution space exploration.
    """

    @staticmethod
    def embedding_space_coverage(
        embeddings: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute coverage as proportion of embedding space occupied.

        Uses grid-based coverage estimation.
        """
        from sklearn.decomposition import PCA

        if len(embeddings) < 2:
            return 0.0

        # Reduce to 2D
        n_components = min(2, embeddings.shape[1])
        pca = PCA(n_components=n_components)

        try:
            reduced = pca.fit_transform(embeddings)
        except Exception:
            return 0.0

        # Normalize to [0, 1]
        min_vals = reduced.min(axis=0)
        max_vals = reduced.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals = np.maximum(range_vals, 1e-10)
        normalized = (reduced - min_vals) / range_vals

        # Count occupied cells
        if n_components == 2:
            cell_indices = (normalized * (n_bins - 1)).astype(int)
            cell_indices = np.clip(cell_indices, 0, n_bins - 1)
            occupied_cells = set(tuple(idx) for idx in cell_indices)
            total_cells = n_bins ** 2
        else:
            cell_indices = (normalized * (n_bins - 1)).astype(int)
            cell_indices = np.clip(cell_indices, 0, n_bins - 1)
            occupied_cells = set(cell_indices.flatten())
            total_cells = n_bins

        coverage = len(occupied_cells) / total_cells

        return float(coverage)

    @staticmethod
    def convex_hull_volume(
        embeddings: np.ndarray,
        n_components: int = 3
    ) -> float:
        """
        Compute volume of convex hull of embeddings.

        Larger volume indicates more diverse/spread responses.
        """
        from sklearn.decomposition import PCA
        from scipy.spatial import ConvexHull

        if len(embeddings) <= n_components:
            return 0.0

        # Reduce dimensionality
        actual_components = min(n_components, embeddings.shape[1])
        pca = PCA(n_components=actual_components)

        try:
            reduced = pca.fit_transform(embeddings)
        except Exception:
            return 0.0

        # Compute convex hull
        try:
            hull = ConvexHull(reduced)
            return float(hull.volume)
        except Exception:
            return 0.0


class PhaseTransitionAnalyzer:
    """
    Analyze phase transition behavior as competitive pressure varies.
    """

    @staticmethod
    def detect_critical_point(
        alpha_values: List[float],
        niche_counts: List[float],
        method: str = "derivative"
    ) -> Dict[str, any]:
        """
        Detect the critical point α_c where phase transition occurs.

        Args:
            alpha_values: List of α values tested
            niche_counts: Corresponding niche counts

        Returns:
            Dictionary with critical point estimate and confidence
        """
        alpha_arr = np.array(alpha_values)
        niche_arr = np.array(niche_counts)

        if method == "derivative":
            # Find maximum derivative (steepest change)
            if len(alpha_arr) < 3:
                return {"alpha_c": None, "confidence": 0.0}

            # Compute derivative
            d_niche = np.diff(niche_arr)
            d_alpha = np.diff(alpha_arr)
            derivative = d_niche / (d_alpha + 1e-10)

            # Find maximum
            max_idx = np.argmax(np.abs(derivative))
            alpha_c = (alpha_arr[max_idx] + alpha_arr[max_idx + 1]) / 2

            # Confidence based on magnitude of derivative
            confidence = np.abs(derivative[max_idx]) / (np.max(niche_arr) + 1e-10)
            confidence = min(confidence, 1.0)

        elif method == "threshold":
            # Find α where niche count first exceeds 1
            threshold = 1.5
            above_threshold = niche_arr > threshold
            if np.any(above_threshold):
                first_idx = np.argmax(above_threshold)
                alpha_c = alpha_arr[first_idx]
                confidence = 0.8
            else:
                alpha_c = None
                confidence = 0.0

        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            "alpha_c": float(alpha_c) if alpha_c is not None else None,
            "confidence": float(confidence),
            "method": method
        }

    @staticmethod
    def fit_scaling_law(
        alpha_values: List[float],
        niche_counts: List[float],
        alpha_c: float
    ) -> Dict[str, any]:
        """
        Fit the scaling law k ~ (α/α_c)^γ.

        Returns:
            Dictionary with fitted parameters
        """
        alpha_arr = np.array(alpha_values)
        niche_arr = np.array(niche_counts)

        # Filter to α > α_c
        mask = alpha_arr > alpha_c
        if np.sum(mask) < 2:
            return {"gamma": None, "r_squared": 0.0}

        alpha_above = alpha_arr[mask]
        niche_above = niche_arr[mask]

        # Log-linear fit: log(k) = γ * log(α/α_c) + const
        x = np.log(alpha_above / alpha_c + 1e-10)
        y = np.log(niche_above + 1e-10)

        # Linear regression
        try:
            coeffs = np.polyfit(x, y, 1)
            gamma = coeffs[0]

            # R-squared
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        except Exception:
            return {"gamma": None, "r_squared": 0.0}

        return {
            "gamma": float(gamma),
            "r_squared": float(r_squared),
            "alpha_c": float(alpha_c)
        }


def compute_all_metrics(
    embeddings: np.ndarray,
    texts: List[str],
    quality_scores: Optional[List[float]] = None
) -> Dict[str, any]:
    """
    Compute all metrics for a set of responses.

    Args:
        embeddings: (N, D) array of response embeddings
        texts: List of response texts
        quality_scores: Optional list of quality scores

    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}

    # Diversity metrics
    diversity = DiversityMetrics.semantic_diversity(embeddings)
    metrics["diversity"] = diversity
    metrics["distinct_2"] = DiversityMetrics.distinct_n(texts, n=2)
    metrics["distinct_3"] = DiversityMetrics.distinct_n(texts, n=3)
    metrics["jsd"] = DiversityMetrics.jensen_shannon_divergence(embeddings)

    # Niche metrics
    niche_dbscan = NicheMetrics.count_niches_dbscan(embeddings)
    niche_hier = NicheMetrics.count_niches_hierarchical(embeddings)
    metrics["niche_dbscan"] = niche_dbscan
    metrics["niche_hierarchical"] = niche_hier

    # Coverage metrics
    metrics["coverage"] = CoverageMetrics.embedding_space_coverage(embeddings)
    metrics["hull_volume"] = CoverageMetrics.convex_hull_volume(embeddings)

    # Quality metrics
    if quality_scores:
        metrics["quality"] = {
            "mean": float(np.mean(quality_scores)),
            "std": float(np.std(quality_scores)),
            "min": float(np.min(quality_scores)),
            "max": float(np.max(quality_scores))
        }

    metrics["n_responses"] = len(embeddings)

    return metrics


class StatisticalTests:
    """
    Statistical significance tests for comparing protocols.
    """

    @staticmethod
    def paired_t_test(
        scores_a: List[float],
        scores_b: List[float]
    ) -> Dict[str, float]:
        """
        Perform paired t-test between two conditions.

        Args:
            scores_a: Scores from condition A
            scores_b: Scores from condition B

        Returns:
            Dictionary with t-statistic, p-value, and effect size
        """
        from scipy import stats

        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have equal length")

        if len(scores_a) < 2:
            return {"t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0}

        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        # Cohen's d for paired samples
        diff = np.array(scores_a) - np.array(scores_b)
        cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d)
        }

    @staticmethod
    def independent_t_test(
        scores_a: List[float],
        scores_b: List[float]
    ) -> Dict[str, float]:
        """
        Perform independent samples t-test between two conditions.

        Args:
            scores_a: Scores from condition A
            scores_b: Scores from condition B

        Returns:
            Dictionary with t-statistic, p-value, and effect size
        """
        from scipy import stats

        if len(scores_a) < 2 or len(scores_b) < 2:
            return {"t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0}

        t_stat, p_value = stats.ttest_ind(scores_a, scores_b)

        # Cohen's d for independent samples
        n_a, n_b = len(scores_a), len(scores_b)
        var_a, var_b = np.var(scores_a, ddof=1), np.var(scores_b, ddof=1)
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        cohens_d = (np.mean(scores_a) - np.mean(scores_b)) / (pooled_std + 1e-10)

        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d)
        }

    @staticmethod
    def wilcoxon_test(
        scores_a: List[float],
        scores_b: List[float]
    ) -> Dict[str, float]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative).

        Args:
            scores_a: Scores from condition A
            scores_b: Scores from condition B

        Returns:
            Dictionary with statistic and p-value
        """
        from scipy import stats

        if len(scores_a) != len(scores_b) or len(scores_a) < 2:
            return {"statistic": 0.0, "p_value": 1.0}

        try:
            statistic, p_value = stats.wilcoxon(scores_a, scores_b)
            return {
                "statistic": float(statistic),
                "p_value": float(p_value)
            }
        except Exception:
            return {"statistic": 0.0, "p_value": 1.0}

    @staticmethod
    def mann_whitney_test(
        scores_a: List[float],
        scores_b: List[float]
    ) -> Dict[str, float]:
        """
        Perform Mann-Whitney U test (non-parametric, independent samples).

        Args:
            scores_a: Scores from condition A
            scores_b: Scores from condition B

        Returns:
            Dictionary with U-statistic and p-value
        """
        from scipy import stats

        if len(scores_a) < 2 or len(scores_b) < 2:
            return {"u_stat": 0.0, "p_value": 1.0}

        u_stat, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')

        return {
            "u_stat": float(u_stat),
            "p_value": float(p_value)
        }

    @staticmethod
    def bootstrap_ci(
        scores: List[float],
        n_bootstrap: int = 10000,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval for the mean.

        Args:
            scores: List of scores
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Dictionary with mean, lower and upper CI bounds
        """
        scores_arr = np.array(scores)
        n = len(scores_arr)

        if n < 2:
            mean = float(scores_arr[0]) if n == 1 else 0.0
            return {"mean": mean, "ci_lower": mean, "ci_upper": mean}

        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            resample = np.random.choice(scores_arr, size=n, replace=True)
            bootstrap_means.append(np.mean(resample))

        bootstrap_means = np.array(bootstrap_means)

        # Compute percentiles
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return {
            "mean": float(np.mean(scores_arr)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std_error": float(np.std(bootstrap_means))
        }

    @staticmethod
    def effect_size_interpretation(cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size.

        Returns interpretation string: negligible, small, medium, large, or very large
        """
        d = abs(cohens_d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        elif d < 1.2:
            return "large"
        else:
            return "very large"


class DownstreamEvaluator:
    """
    Evaluate downstream task performance as a function of diversity.

    Tests whether increased diversity leads to better downstream outcomes.
    """

    @staticmethod
    def diversity_quality_correlation(
        diversity_scores: List[float],
        quality_scores: List[float]
    ) -> Dict[str, float]:
        """
        Compute correlation between diversity and quality.

        Returns Pearson and Spearman correlations.
        """
        from scipy import stats

        if len(diversity_scores) < 3 or len(quality_scores) < 3:
            return {
                "pearson_r": 0.0, "pearson_p": 1.0,
                "spearman_r": 0.0, "spearman_p": 1.0
            }

        pearson_r, pearson_p = stats.pearsonr(diversity_scores, quality_scores)
        spearman_r, spearman_p = stats.spearmanr(diversity_scores, quality_scores)

        return {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p)
        }

    @staticmethod
    def ensemble_accuracy(
        responses: List[Dict],
        reference_answer: str,
        voting_method: str = "majority"
    ) -> Dict[str, float]:
        """
        Evaluate ensemble accuracy using voting.

        Args:
            responses: List of response dicts with 'text' and optionally 'embedding'
            reference_answer: Ground truth answer
            voting_method: 'majority' or 'weighted'

        Returns:
            Dictionary with accuracy metrics
        """
        if not responses:
            return {"accuracy": 0.0, "agreement": 0.0}

        # Simple text matching (in practice, use more sophisticated comparison)
        reference_lower = reference_answer.lower().strip()

        matches = []
        for r in responses:
            response_text = r.get('text', '').lower().strip()
            # Check if key elements match
            match_score = 1.0 if reference_lower in response_text else 0.0
            matches.append(match_score)

        # Majority voting accuracy
        if voting_method == "majority":
            majority_correct = np.mean(matches) >= 0.5
            accuracy = 1.0 if majority_correct else 0.0
        else:
            accuracy = float(np.mean(matches))

        # Agreement among responses
        agreement = 1.0 - np.std(matches) if len(matches) > 1 else 1.0

        return {
            "accuracy": accuracy,
            "agreement": float(agreement),
            "individual_accuracies": matches
        }

    @staticmethod
    def coverage_quality_tradeoff(
        coverage_scores: List[float],
        quality_scores: List[float]
    ) -> Dict[str, float]:
        """
        Analyze the tradeoff between coverage and quality.

        Returns Pareto frontier metrics.
        """
        if len(coverage_scores) < 2:
            return {"pareto_optimal_count": 0, "hypervolume": 0.0}

        # Find Pareto-optimal points
        pareto_optimal = []
        for i, (c, q) in enumerate(zip(coverage_scores, quality_scores)):
            is_dominated = False
            for j, (c2, q2) in enumerate(zip(coverage_scores, quality_scores)):
                if i != j and c2 >= c and q2 >= q and (c2 > c or q2 > q):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_optimal.append(i)

        # Compute hypervolume (simplified 2D version)
        # Reference point at (0, 0)
        pareto_points = [(coverage_scores[i], quality_scores[i]) for i in pareto_optimal]
        pareto_points.sort(key=lambda x: x[0])

        hypervolume = 0.0
        prev_c = 0.0
        for c, q in pareto_points:
            hypervolume += (c - prev_c) * q
            prev_c = c

        return {
            "pareto_optimal_count": len(pareto_optimal),
            "pareto_optimal_indices": pareto_optimal,
            "hypervolume": float(hypervolume)
        }

    @staticmethod
    def best_of_n_quality(
        quality_scores: List[float],
        diversity_scores: List[float]
    ) -> Dict[str, float]:
        """
        Analyze best-of-N selection as function of diversity.

        Tests whether higher diversity populations produce better best responses.
        """
        if len(quality_scores) == 0:
            return {"best_quality": 0.0, "diversity_at_best": 0.0}

        best_idx = np.argmax(quality_scores)

        return {
            "best_quality": float(quality_scores[best_idx]),
            "diversity_at_best": float(diversity_scores[best_idx]) if diversity_scores else 0.0,
            "mean_quality": float(np.mean(quality_scores)),
            "quality_std": float(np.std(quality_scores)),
            "best_vs_mean_ratio": float(quality_scores[best_idx] / (np.mean(quality_scores) + 1e-10))
        }


def compare_protocols_statistically(
    protocol_results: Dict[str, List[Dict]],
    metric_key: str = "diversity.mean_distance"
) -> Dict[str, any]:
    """
    Perform comprehensive statistical comparison of protocols.

    Args:
        protocol_results: Dict mapping protocol name to list of result dicts
        metric_key: Dot-separated path to metric (e.g., "diversity.mean_distance")

    Returns:
        Dictionary with pairwise comparisons and summary statistics
    """

    def extract_metric(result: Dict, key: str) -> float:
        """Extract nested metric value."""
        parts = key.split(".")
        value = result
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return 0.0
        return float(value) if value is not None else 0.0

    # Extract scores for each protocol
    protocol_scores = {}
    for name, results in protocol_results.items():
        protocol_scores[name] = [extract_metric(r, metric_key) for r in results]

    # Summary statistics with bootstrap CI
    summary = {}
    for name, scores in protocol_scores.items():
        ci = StatisticalTests.bootstrap_ci(scores)
        summary[name] = {
            "mean": ci["mean"],
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "n": len(scores)
        }

    # Pairwise comparisons
    pairwise = {}
    protocol_names = list(protocol_scores.keys())

    for i, name_a in enumerate(protocol_names):
        for name_b in protocol_names[i + 1:]:
            comparison_key = f"{name_a}_vs_{name_b}"

            t_test = StatisticalTests.independent_t_test(
                protocol_scores[name_a],
                protocol_scores[name_b]
            )
            mann_whitney = StatisticalTests.mann_whitney_test(
                protocol_scores[name_a],
                protocol_scores[name_b]
            )

            pairwise[comparison_key] = {
                "t_test": t_test,
                "mann_whitney": mann_whitney,
                "effect_size": t_test["cohens_d"],
                "effect_interpretation": StatisticalTests.effect_size_interpretation(
                    t_test["cohens_d"]
                ),
                "significant_005": t_test["p_value"] < 0.05,
                "significant_001": t_test["p_value"] < 0.01
            }

    return {
        "metric": metric_key,
        "summary": summary,
        "pairwise_comparisons": pairwise
    }


if __name__ == "__main__":
    # Test metrics computation
    print("Testing Metrics Computation")
    print("=" * 50)

    np.random.seed(42)

    # Create synthetic data
    n_samples = 20
    embedding_dim = 64

    # Clustered embeddings (simulate niche formation)
    n_clusters = 3
    embeddings = []
    for c in range(n_clusters):
        center = np.random.randn(embedding_dim)
        center = center / np.linalg.norm(center)
        for _ in range(n_samples // n_clusters):
            point = center + 0.1 * np.random.randn(embedding_dim)
            point = point / np.linalg.norm(point)
            embeddings.append(point)

    embeddings = np.array(embeddings)
    texts = [f"Sample response {i}" for i in range(len(embeddings))]
    quality_scores = np.random.uniform(0.5, 1.0, len(embeddings)).tolist()

    # Compute all metrics
    metrics = compute_all_metrics(embeddings, texts, quality_scores)

    print(f"\nDiversity Metrics:")
    print(f"  Mean distance: {metrics['diversity']['mean_distance']:.4f}")
    print(f"  Distinct-2: {metrics['distinct_2']:.4f}")
    print(f"  JSD: {metrics['jsd']:.4f}")

    print(f"\nNiche Metrics:")
    print(f"  DBSCAN clusters: {metrics['niche_dbscan']['n_niches']}")
    print(f"  Hierarchical clusters: {metrics['niche_hierarchical']['n_niches']}")

    print(f"\nCoverage Metrics:")
    print(f"  Grid coverage: {metrics['coverage']:.4f}")
    print(f"  Hull volume: {metrics['hull_volume']:.4f}")

    print(f"\nQuality Metrics:")
    print(f"  Mean quality: {metrics['quality']['mean']:.4f}")

"""
Semantic Mapping for Emergent Niche Interpretation

Implements the methodology from:
"LLM-Based Semantic Mapping for Structured Conceptual Exploration" (NBiS 2025)

This module provides tools to:
1. Extract semantic features from LLM responses
2. Apply PCA for dimensionality reduction
3. Generate interpretable axis labels via LLM
4. Visualize emergent niches in 2D semantic space

Key contribution: Unlike MAP-Elites which requires pre-defined axes,
this approach discovers and interprets axes POST-HOC from emergent niches.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

from .llm_agent import get_openai_client, get_anthropic_client


@dataclass
class SemanticFeature:
    """A semantic feature extracted from responses."""
    name: str
    description: str
    scores: Dict[int, float]  # agent_id -> score


@dataclass
class SemanticAxis:
    """An interpreted axis from PCA."""
    axis_id: int  # 0 for PC1, 1 for PC2
    label: str
    positive_pole: str  # Description of high values
    negative_pole: str  # Description of low values
    explained_variance: float
    loadings: Dict[str, float]  # feature_name -> loading


@dataclass
class SemanticMap:
    """Complete semantic map with coordinates and axes."""
    coordinates: np.ndarray  # (n_responses, 2)
    axes: List[SemanticAxis]
    feature_matrix: np.ndarray  # (n_responses, n_features)
    feature_names: List[str]
    response_ids: List[int]


class SemanticFeatureExtractor:
    """
    Extracts semantic features from text responses using LLM.

    Based on NBiS 2025 methodology:
    - LLM extracts emotionally resonant, human-interpretable features
    - Each response is scored on extracted features (0.0-1.0)
    """

    SYSTEM_PROMPT = """You are a perceptive assistant that analyzes meaning and extracts emotionally resonant and human-interpretable features. Avoid generic or overly technical labels. Instead, choose expressive, vivid features that reflect how humans intuitively perceive or feel about each item."""

    EXTRACTION_PROMPT_TEMPLATE = """From the following list of responses, extract {n_features} emotionally expressive and specific features that are commonly applicable to all items. Each feature should reflect how people might intuitively experience or describe the responses (e.g., "Boldness", "Analytical Depth", "Creative Flair", "Pragmatic Focus").

For each response, rate how strongly it exhibits each feature on a scale from 0.0 to 1.0.
Make the scores as polarized as possible to highlight contrast between items.

Output must be in the following JSON format. Do not include any explanation.
{{
  "features": ["Feature1", "Feature2", ...],
  "items": [
    {{
      "id": 0,
      "scores": {{"Feature1": 0.9, "Feature2": 0.2, ...}}
    }},
    ...
  ]
}}

Responses to analyze:
{responses}"""

    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4o-mini",
        n_features: int = 8
    ):
        self.backend = backend
        self.model = model
        self.n_features = n_features

    def extract_features(
        self,
        responses: List[Tuple[int, str]]  # List of (id, text)
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract semantic features from responses.

        Returns:
            feature_matrix: (n_responses, n_features) array of scores
            feature_names: List of feature names
        """
        # Format responses for prompt
        response_text = "\n\n".join([
            f"[Response {id}]: {text[:500]}..."  # Truncate long responses
            if len(text) > 500 else f"[Response {id}]: {text}"
            for id, text in responses
        ])

        prompt = self.EXTRACTION_PROMPT_TEMPLATE.format(
            n_features=self.n_features,
            responses=response_text
        )

        # Call LLM
        if self.backend == "openai":
            result = self._call_openai(prompt)
        elif self.backend == "anthropic":
            result = self._call_anthropic(prompt)
        else:
            # Mock: generate random features (return directly, no JSON parsing)
            return self._mock_extraction(responses)

        # Parse result
        try:
            data = json.loads(result)
            feature_names = data["features"]

            # Build feature matrix
            id_to_idx = {id: idx for idx, (id, _) in enumerate(responses)}
            n_responses = len(responses)
            n_features = len(feature_names)

            feature_matrix = np.zeros((n_responses, n_features))
            for item in data["items"]:
                idx = id_to_idx.get(item["id"], item["id"])
                for j, feat in enumerate(feature_names):
                    feature_matrix[idx, j] = item["scores"].get(feat, 0.5)

            return feature_matrix, feature_names

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            return self._mock_extraction(responses)

    def _call_openai(self, prompt: str) -> str:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str) -> str:
        client = get_anthropic_client()
        response = client.messages.create(
            model=self.model,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        return response.content[0].text

    def _mock_extraction(
        self,
        responses: List[Tuple[int, str]]
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate mock features for testing."""
        feature_names = [
            "Analytical Depth", "Creative Flair", "Pragmatic Focus",
            "Emotional Resonance", "Technical Rigor", "Novelty",
            "Clarity", "Comprehensiveness"
        ][:self.n_features]

        n_responses = len(responses)
        # Generate deterministic but varied scores based on response length
        feature_matrix = np.zeros((n_responses, self.n_features))
        for i, (id, text) in enumerate(responses):
            np.random.seed(hash(text) % (2**32))
            feature_matrix[i] = np.random.uniform(0.2, 0.8, self.n_features)

        return feature_matrix, feature_names


class SemanticAxisLabeler:
    """
    Generates interpretable labels for PCA axes using LLM.

    Based on NBiS 2025 methodology:
    - Takes PCA loadings as input
    - Generates concise, intuitive axis labels
    """

    LABELING_PROMPT_TEMPLATE = """You are given a list of features with numerical weights (PCA loadings). These loadings indicate how much each feature contributes to a principal component axis.

Positive loadings mean the feature increases along the axis.
Negative loadings mean the feature decreases along the axis.

For each axis, provide:
1. A short label (max 20 characters) that captures the essence
2. What HIGH values on this axis represent
3. What LOW values on this axis represent

PCA Loadings:

Axis 1 (explains {var1:.1%} of variance):
{loadings1}

Axis 2 (explains {var2:.1%} of variance):
{loadings2}

Output in JSON format:
{{
  "axis1": {{
    "label": "Short Label",
    "high": "Description of high values",
    "low": "Description of low values"
  }},
  "axis2": {{
    "label": "Short Label",
    "high": "Description of high values",
    "low": "Description of low values"
  }}
}}"""

    def __init__(self, backend: str = "openai", model: str = "gpt-4o-mini"):
        self.backend = backend
        self.model = model

    def generate_labels(
        self,
        loadings: np.ndarray,  # (n_features, 2)
        feature_names: List[str],
        explained_variance: np.ndarray  # (2,)
    ) -> List[SemanticAxis]:
        """Generate interpretable labels for PCA axes."""

        # Format loadings for prompt
        def format_loadings(axis_idx: int) -> str:
            pairs = [(name, loadings[i, axis_idx])
                     for i, name in enumerate(feature_names)]
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            return "\n".join([f"  {name}: {val:+.3f}" for name, val in pairs])

        prompt = self.LABELING_PROMPT_TEMPLATE.format(
            var1=explained_variance[0],
            var2=explained_variance[1],
            loadings1=format_loadings(0),
            loadings2=format_loadings(1)
        )

        # Call LLM
        if self.backend == "openai":
            result = self._call_openai(prompt)
        elif self.backend == "anthropic":
            result = self._call_anthropic(prompt)
        else:
            result = self._mock_labels()

        # Parse result
        try:
            data = json.loads(result)
            axes = []
            for i, key in enumerate(["axis1", "axis2"]):
                axis_data = data[key]
                axes.append(SemanticAxis(
                    axis_id=i,
                    label=axis_data["label"],
                    positive_pole=axis_data["high"],
                    negative_pole=axis_data["low"],
                    explained_variance=float(explained_variance[i]),
                    loadings={name: float(loadings[j, i])
                              for j, name in enumerate(feature_names)}
                ))
            return axes
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse axis labels: {e}")
            return self._default_axes(loadings, feature_names, explained_variance)

    def _call_openai(self, prompt: str) -> str:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str) -> str:
        client = get_anthropic_client()
        response = client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.content[0].text

    def _mock_labels(self) -> str:
        return json.dumps({
            "axis1": {
                "label": "Analytical-Creative",
                "high": "Highly analytical and structured",
                "low": "Creative and exploratory"
            },
            "axis2": {
                "label": "Abstract-Concrete",
                "high": "Abstract and theoretical",
                "low": "Concrete and practical"
            }
        })

    def _default_axes(
        self,
        loadings: np.ndarray,
        feature_names: List[str],
        explained_variance: np.ndarray
    ) -> List[SemanticAxis]:
        """Generate default axes without LLM."""
        axes = []
        for i in range(2):
            # Find dominant features
            sorted_idx = np.argsort(np.abs(loadings[:, i]))[::-1]
            top_positive = [feature_names[j] for j in sorted_idx[:2]
                           if loadings[j, i] > 0]
            top_negative = [feature_names[j] for j in sorted_idx[:2]
                           if loadings[j, i] < 0]

            label = f"PC{i+1}"
            if top_positive:
                label = top_positive[0][:15]

            axes.append(SemanticAxis(
                axis_id=i,
                label=label,
                positive_pole=", ".join(top_positive) if top_positive else "High",
                negative_pole=", ".join(top_negative) if top_negative else "Low",
                explained_variance=float(explained_variance[i]),
                loadings={name: float(loadings[j, i])
                          for j, name in enumerate(feature_names)}
            ))
        return axes


class SemanticMapper:
    """
    Complete semantic mapping pipeline.

    Integrates:
    1. Feature extraction
    2. PCA projection
    3. Axis labeling
    4. Coordinate generation
    """

    def __init__(
        self,
        backend: str = "mock",
        model: str = "gpt-4o-mini",
        n_features: int = 8
    ):
        self.feature_extractor = SemanticFeatureExtractor(
            backend=backend, model=model, n_features=n_features
        )
        self.axis_labeler = SemanticAxisLabeler(backend=backend, model=model)

    def create_semantic_map(
        self,
        responses: List[Tuple[int, str]]  # List of (id, text)
    ) -> SemanticMap:
        """
        Create a complete semantic map from responses.

        This is the key method for interpreting emergent niches.
        """
        from sklearn.decomposition import PCA

        # Step 1: Extract semantic features
        feature_matrix, feature_names = self.feature_extractor.extract_features(responses)

        # Step 2: Center the data
        centered = feature_matrix - feature_matrix.mean(axis=0)

        # Step 3: PCA projection
        pca = PCA(n_components=2)
        coordinates = pca.fit_transform(centered)

        # Get loadings (eigenvectors scaled by sqrt of eigenvalues)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        # Step 4: Generate axis labels
        axes = self.axis_labeler.generate_labels(
            loadings=loadings,
            feature_names=feature_names,
            explained_variance=pca.explained_variance_ratio_
        )

        return SemanticMap(
            coordinates=coordinates,
            axes=axes,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            response_ids=[id for id, _ in responses]
        )

    def interpret_clusters(
        self,
        semantic_map: SemanticMap,
        cluster_labels: np.ndarray
    ) -> Dict[int, Dict]:
        """
        Interpret what each cluster represents in semantic terms.

        This enables POST-HOC interpretation of emergent niches from DDS.
        """
        unique_labels = np.unique(cluster_labels)
        interpretations = {}

        for label in unique_labels:
            mask = cluster_labels == label
            cluster_coords = semantic_map.coordinates[mask]
            cluster_features = semantic_map.feature_matrix[mask]

            # Centroid position
            centroid = cluster_coords.mean(axis=0)

            # Dominant features in this cluster
            mean_features = cluster_features.mean(axis=0)
            sorted_idx = np.argsort(mean_features)[::-1]

            top_features = [
                (semantic_map.feature_names[i], float(mean_features[i]))
                for i in sorted_idx[:3]
            ]

            # Position interpretation
            axis1_pos = "high" if centroid[0] > 0 else "low"
            axis2_pos = "high" if centroid[1] > 0 else "low"

            interpretations[int(label)] = {
                "n_members": int(mask.sum()),
                "centroid": centroid.tolist(),
                "top_features": top_features,
                "axis1_interpretation": f"{axis1_pos} {semantic_map.axes[0].label}",
                "axis2_interpretation": f"{axis2_pos} {semantic_map.axes[1].label}",
                "description": f"Cluster characterized by {top_features[0][0]} ({top_features[0][1]:.2f})"
            }

        return interpretations


def analyze_emergent_niches(
    responses: List[Tuple[int, str]],
    cluster_labels: np.ndarray,
    backend: str = "mock"
) -> Dict:
    """
    High-level function to analyze emergent niches from DDS.

    This is the key differentiator from MAP-Elites:
    - MAP-Elites: axes defined a priori
    - DDS + Semantic Mapping: axes discovered post-hoc

    Returns a complete analysis including:
    - Semantic map with coordinates
    - Interpreted axes
    - Cluster interpretations
    """
    mapper = SemanticMapper(backend=backend)

    # Create semantic map
    semantic_map = mapper.create_semantic_map(responses)

    # Interpret clusters
    cluster_interpretations = mapper.interpret_clusters(semantic_map, cluster_labels)

    return {
        "coordinates": semantic_map.coordinates.tolist(),
        "axes": [
            {
                "label": axis.label,
                "positive_pole": axis.positive_pole,
                "negative_pole": axis.negative_pole,
                "explained_variance": axis.explained_variance
            }
            for axis in semantic_map.axes
        ],
        "feature_names": semantic_map.feature_names,
        "cluster_interpretations": cluster_interpretations,
        "total_explained_variance": sum(ax.explained_variance for ax in semantic_map.axes)
    }


if __name__ == "__main__":
    print("Testing Semantic Mapping")
    print("=" * 50)

    # Create test responses
    test_responses = [
        (0, "The solution focuses on practical implementation with clear steps and measurable outcomes."),
        (1, "An innovative approach that challenges conventional thinking with creative abstractions."),
        (2, "A balanced perspective considering both theoretical foundations and real-world applications."),
        (3, "Highly technical analysis with rigorous methodology and quantitative evidence."),
        (4, "Emphasizes human-centered design with emotional intelligence and user empathy."),
    ]

    # Test with mock backend
    mapper = SemanticMapper(backend="mock", n_features=6)
    semantic_map = mapper.create_semantic_map(test_responses)

    print(f"\nSemantic Map Created:")
    print(f"  Coordinates shape: {semantic_map.coordinates.shape}")
    print(f"  Features: {semantic_map.feature_names}")
    print(f"\nAxes:")
    for axis in semantic_map.axes:
        print(f"  {axis.axis_id}: {axis.label}")
        print(f"    High: {axis.positive_pole}")
        print(f"    Low: {axis.negative_pole}")
        print(f"    Variance: {axis.explained_variance:.1%}")

    # Test cluster interpretation
    cluster_labels = np.array([0, 1, 0, 1, 0])
    interpretations = mapper.interpret_clusters(semantic_map, cluster_labels)

    print(f"\nCluster Interpretations:")
    for label, interp in interpretations.items():
        print(f"  Cluster {label}: {interp['description']}")

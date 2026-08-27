"""
Multi-Agent Interaction Protocols

Implements various protocols for multi-agent interaction:
1. Independent: No interaction between agents
2. Debate: Agents see and respond to each other's responses
3. Diversity Prompt: Agents instructed to be diverse
4. Role Assignment: Agents have predefined roles
5. Density-Dependent Selection (DDS): Our proposed method
6. Fitness Sharing: Traditional EA baseline
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from copy import deepcopy

from .llm_agent import (
    LLMAgent, MultiAgentSystem, Task, Response,
    AgentConfig, create_agent, EmbeddingComputer
)
from .density_selection import (
    DensityDependentSelector, SelectionConfig,
    AgentResponse, FitnessSharingSelector
)


@dataclass
class ProtocolConfig:
    """Base configuration for protocols."""
    n_rounds: int = 3              # Number of interaction rounds
    n_agents: int = 5              # Number of agents


@dataclass
class DebateConfig(ProtocolConfig):
    """Configuration for debate protocol."""
    show_all_responses: bool = True    # Show all responses vs. summary


@dataclass
class DiversityPromptConfig(ProtocolConfig):
    """Configuration for diversity prompt protocol."""
    diversity_instruction: str = "Please provide a unique and different perspective from others."


@dataclass
class RoleConfig(ProtocolConfig):
    """Configuration for role-based protocol."""
    roles: List[str] = field(default_factory=lambda: [
        "Critical Analyst: Focus on identifying potential flaws and limitations.",
        "Creative Thinker: Propose innovative and unconventional solutions.",
        "Pragmatist: Consider practical implementation and feasibility.",
        "Devil's Advocate: Challenge assumptions and present counterarguments.",
        "Synthesizer: Integrate different perspectives into a coherent whole."
    ])


@dataclass
class DDSConfig(ProtocolConfig):
    """Configuration for density-dependent selection protocol."""
    alpha: float = 1.0             # Competitive pressure
    beta: float = 2.0              # Selection temperature
    bandwidth: float = 0.3         # Kernel bandwidth
    n_survive: int = 3             # Agents to survive each round


@dataclass
class FitnessSharingConfig(ProtocolConfig):
    """Configuration for fitness sharing protocol."""
    share_radius: float = 0.5      # Sharing radius
    alpha: float = 1.0             # Sharing power
    n_survive: int = 3             # Agents to survive each round


@dataclass
class DebateDDSConfig(ProtocolConfig):
    """Configuration for Debate+DDS hybrid protocol."""
    alpha: float = 0.5             # Competitive pressure
    beta: float = 2.0              # Selection temperature
    bandwidth: float = 0.3         # Kernel bandwidth
    n_survive: int = 3             # Agents to survive each round


@dataclass
class NoveltySearchConfig(ProtocolConfig):
    """Configuration for novelty search protocol (Lehman & Stanley 2011)."""
    k_nearest: int = 15            # Number of nearest neighbors for novelty calculation
    archive_prob: float = 0.1     # Probability of adding to archive
    archive_threshold: float = 0.3  # Novelty threshold for archive addition
    n_survive: int = 3             # Agents to survive each round


class Protocol(ABC):
    """Abstract base class for multi-agent protocols."""

    def __init__(self, config: ProtocolConfig):
        self.config = config

    @abstractmethod
    def run(
        self,
        system: MultiAgentSystem,
        task: Task,
        quality_evaluator: Optional[Callable] = None
    ) -> Dict:
        """
        Run the protocol on a task.

        Returns:
            Dictionary containing:
            - final_responses: List of final responses
            - round_history: History of each round
            - metrics: Protocol-specific metrics
        """
        pass


class IndependentProtocol(Protocol):
    """
    Independent protocol: Agents generate responses without any interaction.
    This serves as the diversity upper bound (no convergence pressure).
    """

    def run(
        self,
        system: MultiAgentSystem,
        task: Task,
        quality_evaluator: Optional[Callable] = None
    ) -> Dict:
        system.reset_all()

        # Generate responses independently (only one round makes sense)
        responses = system.generate_responses(task)

        # Evaluate quality if evaluator provided
        if quality_evaluator:
            for r in responses:
                r.quality_score = quality_evaluator(r.text, task)

        return {
            "final_responses": responses,
            "round_history": [responses],
            "metrics": {
                "protocol": "independent",
                "n_rounds": 1
            }
        }


class DebateProtocol(Protocol):
    """
    Standard debate protocol: Agents see each other's responses and can respond.
    This typically leads to convergence (diversity collapse).
    """

    def __init__(self, config: DebateConfig):
        super().__init__(config)

    def run(
        self,
        system: MultiAgentSystem,
        task: Task,
        quality_evaluator: Optional[Callable] = None
    ) -> Dict:
        system.reset_all()
        round_history = []

        # Initial round
        responses = system.generate_responses(task)
        round_history.append(deepcopy(responses))

        # Subsequent debate rounds
        for round_idx in range(1, self.config.n_rounds):
            # Build context from previous responses
            def context_provider(agent_id: int) -> str:
                other_responses = [
                    f"Agent {r.agent_id}: {r.text}"
                    for r in responses if r.agent_id != agent_id
                ]
                return "\n\n".join(other_responses)

            # Generate new responses with context
            debate_prompt = f"{task.prompt}\n\nConsider the perspectives shared and provide your response."
            debate_task = Task(
                task_id=f"{task.task_id}_round{round_idx}",
                prompt=debate_prompt,
                category=task.category
            )

            responses = system.generate_responses(debate_task, context_provider)
            round_history.append(deepcopy(responses))

        # Evaluate quality if evaluator provided
        if quality_evaluator:
            for r in responses:
                r.quality_score = quality_evaluator(r.text, task)

        return {
            "final_responses": responses,
            "round_history": round_history,
            "metrics": {
                "protocol": "debate",
                "n_rounds": self.config.n_rounds
            }
        }


class DiversityPromptProtocol(Protocol):
    """
    Diversity prompt protocol: Agents are explicitly instructed to be diverse.
    Based on Paper 2's "Diversity Prompt" condition.
    """

    def __init__(self, config: DiversityPromptConfig):
        super().__init__(config)

    def run(
        self,
        system: MultiAgentSystem,
        task: Task,
        quality_evaluator: Optional[Callable] = None
    ) -> Dict:
        system.reset_all()
        round_history = []

        # Initial round
        responses = system.generate_responses(task)
        round_history.append(deepcopy(responses))

        # Subsequent rounds with diversity instruction
        for round_idx in range(1, self.config.n_rounds):
            def context_provider(agent_id: int) -> str:
                other_responses = [
                    f"Agent {r.agent_id}: {r.text}"
                    for r in responses if r.agent_id != agent_id
                ]
                context = "\n\n".join(other_responses)
                return f"{context}\n\n{self.config.diversity_instruction}"

            diversity_prompt = f"{task.prompt}\n\nImportant: {self.config.diversity_instruction}"
            diversity_task = Task(
                task_id=f"{task.task_id}_round{round_idx}",
                prompt=diversity_prompt,
                category=task.category
            )

            responses = system.generate_responses(diversity_task, context_provider)
            round_history.append(deepcopy(responses))

        if quality_evaluator:
            for r in responses:
                r.quality_score = quality_evaluator(r.text, task)

        return {
            "final_responses": responses,
            "round_history": round_history,
            "metrics": {
                "protocol": "diversity_prompt",
                "n_rounds": self.config.n_rounds
            }
        }


class RoleAssignmentProtocol(Protocol):
    """
    Role assignment protocol: Agents have predefined roles.
    Similar to MetaGPT approach.
    """

    def __init__(self, config: RoleConfig):
        super().__init__(config)

    def run(
        self,
        system: MultiAgentSystem,
        task: Task,
        quality_evaluator: Optional[Callable] = None
    ) -> Dict:
        system.reset_all()
        round_history = []

        # Assign roles to agents
        roles = self.config.roles
        for i, agent in enumerate(system.agents):
            role = roles[i % len(roles)]
            agent.config.system_prompt = f"You are assigned the role of: {role}\nAlways respond from this perspective."

        # Initial round
        responses = system.generate_responses(task)
        round_history.append(deepcopy(responses))

        # Subsequent rounds
        for round_idx in range(1, self.config.n_rounds):
            def context_provider(agent_id: int) -> str:
                other_responses = [
                    f"Agent {r.agent_id}: {r.text}"
                    for r in responses if r.agent_id != agent_id
                ]
                return "\n\n".join(other_responses)

            role_task = Task(
                task_id=f"{task.task_id}_round{round_idx}",
                prompt=f"{task.prompt}\n\nRespond according to your assigned role.",
                category=task.category
            )

            responses = system.generate_responses(role_task, context_provider)
            round_history.append(deepcopy(responses))

        if quality_evaluator:
            for r in responses:
                r.quality_score = quality_evaluator(r.text, task)

        return {
            "final_responses": responses,
            "round_history": round_history,
            "metrics": {
                "protocol": "role_assignment",
                "n_rounds": self.config.n_rounds
            }
        }


class DDSProtocol(Protocol):
    """
    Density-Dependent Selection protocol: Our proposed method.

    Key mechanism: After each round, agents are selected based on
    fitness with density penalty. This promotes niche formation.
    """

    def __init__(self, config: DDSConfig):
        super().__init__(config)
        self.selector = DensityDependentSelector(SelectionConfig(
            alpha=config.alpha,
            beta=config.beta,
            bandwidth=config.bandwidth
        ))

    def run(
        self,
        system: MultiAgentSystem,
        task: Task,
        quality_evaluator: Optional[Callable] = None
    ) -> Dict:
        system.reset_all()
        round_history = []
        selection_history = []

        # Initial round
        responses = system.generate_responses(task)

        # Evaluate quality
        if quality_evaluator:
            for r in responses:
                r.quality_score = quality_evaluator(r.text, task)
        else:
            # Default: fixed quality to isolate diversity mechanism
            for r in responses:
                r.quality_score = 0.75  # Fixed quality to isolate diversity mechanism

        round_history.append(deepcopy(responses))

        # Subsequent rounds with density-dependent selection
        for round_idx in range(1, self.config.n_rounds):
            # Convert to AgentResponse for selection
            agent_responses = [
                AgentResponse(
                    agent_id=r.agent_id,
                    response_text=r.text,
                    embedding=r.embedding,
                    quality_score=r.quality_score,
                    generation=round_idx - 1
                )
                for r in responses
            ]

            # Perform density-dependent selection
            selected, probs = self.selector.select(
                agent_responses,
                n_select=self.config.n_survive
            )

            selection_history.append({
                "round": round_idx,
                "selection_probs": probs.tolist(),
                "fitness": [ar.fitness for ar in agent_responses],
                "density": [ar.local_density for ar in agent_responses]
            })

            # Build context from selected responses
            def context_provider(agent_id: int) -> str:
                selected_texts = [
                    f"Selected response: {s.response_text}"
                    for s in selected
                ]
                return "\n\n".join(selected_texts)

            # Generate new responses
            dds_task = Task(
                task_id=f"{task.task_id}_round{round_idx}",
                prompt=f"{task.prompt}\n\nBuild upon or differentiate from the context.",
                category=task.category
            )

            responses = system.generate_responses(dds_task, context_provider)

            # Evaluate quality
            if quality_evaluator:
                for r in responses:
                    r.quality_score = quality_evaluator(r.text, task)
            else:
                for r in responses:
                    r.quality_score = 0.75  # Fixed quality to isolate diversity mechanism

            round_history.append(deepcopy(responses))

        return {
            "final_responses": responses,
            "round_history": round_history,
            "selection_history": selection_history,
            "metrics": {
                "protocol": "dds",
                "n_rounds": self.config.n_rounds,
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "bandwidth": self.config.bandwidth
            }
        }


class DebateDDSProtocol(Protocol):
    """
    Debate+DDS hybrid protocol.

    Combines Debate's interaction (agents see ALL responses) with
    DDS's density-dependent selection (redundant responses eliminated).

    Round 0: All agents generate initial responses
    Round 1+: Show ALL responses to each agent (Debate-style)
              -> All agents regenerate
              -> Apply density-dependent selection (DDS-style)
              -> Survivors go to next round
    """

    def __init__(self, config: DebateDDSConfig):
        super().__init__(config)
        self.selector = DensityDependentSelector(SelectionConfig(
            alpha=config.alpha,
            beta=config.beta,
            bandwidth=config.bandwidth
        ))

    def run(
        self,
        system: MultiAgentSystem,
        task: Task,
        quality_evaluator: Optional[Callable] = None
    ) -> Dict:
        system.reset_all()
        round_history = []
        selection_history = []

        # Initial round
        responses = system.generate_responses(task)

        # Evaluate quality
        if quality_evaluator:
            for r in responses:
                r.quality_score = quality_evaluator(r.text, task)
        else:
            for r in responses:
                r.quality_score = 0.75

        round_history.append(deepcopy(responses))

        # Subsequent rounds: Debate interaction + DDS selection
        for round_idx in range(1, self.config.n_rounds):
            # Debate-style context: show ALL responses to each agent
            def context_provider(agent_id: int) -> str:
                other_responses = [
                    f"Agent {r.agent_id}: {r.text}"
                    for r in responses if r.agent_id != agent_id
                ]
                return "\n\n".join(other_responses)

            # Generate new responses with debate context
            debate_task = Task(
                task_id=f"{task.task_id}_round{round_idx}",
                prompt=f"{task.prompt}\n\nConsider the perspectives shared and provide your response.",
                category=task.category
            )

            responses = system.generate_responses(debate_task, context_provider)

            # Evaluate quality
            if quality_evaluator:
                for r in responses:
                    r.quality_score = quality_evaluator(r.text, task)
            else:
                for r in responses:
                    r.quality_score = 0.75

            # DDS-style selection: eliminate redundant responses
            agent_responses = [
                AgentResponse(
                    agent_id=r.agent_id,
                    response_text=r.text,
                    embedding=r.embedding,
                    quality_score=r.quality_score,
                    generation=round_idx
                )
                for r in responses
            ]

            selected, probs = self.selector.select(
                agent_responses,
                n_select=self.config.n_survive
            )

            selection_history.append({
                "round": round_idx,
                "selection_probs": probs.tolist(),
                "fitness": [ar.fitness for ar in agent_responses],
                "density": [ar.local_density for ar in agent_responses]
            })

            # Keep all responses in round_history (pre-selection)
            round_history.append(deepcopy(responses))

        return {
            "final_responses": responses,
            "round_history": round_history,
            "selection_history": selection_history,
            "metrics": {
                "protocol": "debate_dds",
                "n_rounds": self.config.n_rounds,
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "bandwidth": self.config.bandwidth
            }
        }


class NoveltySearchProtocol(Protocol):
    """
    Novelty Search protocol: Baseline from Lehman & Stanley (2011).

    Key mechanism: Selection is based purely on novelty (distance from
    nearest neighbors in archive + current population). Quality is ignored.
    This serves as a diversity-only baseline.
    """

    def __init__(self, config: NoveltySearchConfig):
        super().__init__(config)
        self.archive: List[np.ndarray] = []  # Archive of novel responses

    def _compute_novelty(
        self,
        embedding: np.ndarray,
        population_embeddings: List[np.ndarray]
    ) -> float:
        """Compute novelty as mean distance to k nearest neighbors."""
        # Combine archive and current population
        all_embeddings = self.archive + population_embeddings

        if len(all_embeddings) < 2:
            return 1.0  # Maximum novelty if no comparisons available

        # Compute distances
        all_emb_array = np.array(all_embeddings)
        distances = np.array([
            1 - np.dot(embedding, other) / (
                np.linalg.norm(embedding) * np.linalg.norm(other) + 1e-10
            )
            for other in all_emb_array
        ])

        # Get k nearest neighbors (excluding self if present)
        k = min(self.config.k_nearest, len(distances))
        nearest_distances = np.sort(distances)[:k]

        return float(np.mean(nearest_distances))

    def run(
        self,
        system: MultiAgentSystem,
        task: Task,
        quality_evaluator: Optional[Callable] = None
    ) -> Dict:
        system.reset_all()
        self.archive = []  # Reset archive
        round_history = []
        novelty_history = []

        # Initial round
        responses = system.generate_responses(task)
        round_history.append(deepcopy(responses))

        # Subsequent rounds with novelty search
        for round_idx in range(1, self.config.n_rounds):
            # Compute novelty for each response
            population_embeddings = [r.embedding for r in responses]
            novelties = []

            for r in responses:
                novelty = self._compute_novelty(r.embedding, population_embeddings)
                novelties.append(novelty)
                r.quality_score = novelty  # Use novelty as score

            novelty_history.append({
                "round": round_idx,
                "novelties": novelties,
                "archive_size": len(self.archive)
            })

            # Add novel responses to archive (probabilistic or threshold)
            for r, novelty in zip(responses, novelties):
                if novelty > self.config.archive_threshold or \
                   np.random.random() < self.config.archive_prob:
                    self.archive.append(r.embedding.copy())

            # Select based on novelty
            novelty_arr = np.array(novelties)
            # Softmax selection
            probs = np.exp(novelty_arr * 5.0)  # Temperature scaling
            probs = probs / probs.sum()

            selected_indices = np.random.choice(
                len(responses),
                size=min(self.config.n_survive, len(responses)),
                replace=False,
                p=probs
            )
            selected = [responses[i] for i in selected_indices]

            # Build context from selected responses
            def context_provider(agent_id: int) -> str:
                selected_texts = [
                    f"Novel response: {s.text}"
                    for s in selected
                ]
                return "\n\n".join(selected_texts)

            # Generate new responses
            ns_task = Task(
                task_id=f"{task.task_id}_round{round_idx}",
                prompt=f"{task.prompt}\n\nTry to generate a novel and different approach.",
                category=task.category
            )

            responses = system.generate_responses(ns_task, context_provider)
            round_history.append(deepcopy(responses))

        # Final quality evaluation if evaluator provided
        if quality_evaluator:
            for r in responses:
                r.quality_score = quality_evaluator(r.text, task)

        return {
            "final_responses": responses,
            "round_history": round_history,
            "novelty_history": novelty_history,
            "final_archive_size": len(self.archive),
            "metrics": {
                "protocol": "novelty_search",
                "n_rounds": self.config.n_rounds,
                "k_nearest": self.config.k_nearest,
                "archive_threshold": self.config.archive_threshold
            }
        }


class FitnessSharingProtocol(Protocol):
    """
    Fitness Sharing protocol: Traditional EA baseline (Goldberg 1987).

    Similar to DDS but uses multiplicative fitness sharing instead
    of additive density penalty.
    """

    def __init__(self, config: FitnessSharingConfig):
        super().__init__(config)
        selection_config = SelectionConfig(
            alpha=config.alpha,
            beta=2.0,
            bandwidth=config.share_radius
        )
        self.selector = FitnessSharingSelector(
            selection_config,
            share_radius=config.share_radius
        )

    def run(
        self,
        system: MultiAgentSystem,
        task: Task,
        quality_evaluator: Optional[Callable] = None
    ) -> Dict:
        system.reset_all()
        round_history = []
        selection_history = []

        # Initial round
        responses = system.generate_responses(task)

        # Evaluate quality
        if quality_evaluator:
            for r in responses:
                r.quality_score = quality_evaluator(r.text, task)
        else:
            for r in responses:
                r.quality_score = 0.75  # Fixed quality to isolate diversity mechanism

        round_history.append(deepcopy(responses))

        # Subsequent rounds with fitness sharing
        for round_idx in range(1, self.config.n_rounds):
            # Convert to AgentResponse for selection
            agent_responses = [
                AgentResponse(
                    agent_id=r.agent_id,
                    response_text=r.text,
                    embedding=r.embedding,
                    quality_score=r.quality_score,
                    generation=round_idx - 1
                )
                for r in responses
            ]

            # Perform fitness sharing selection
            selected, probs = self.selector.select(
                agent_responses,
                n_select=self.config.n_survive
            )

            selection_history.append({
                "round": round_idx,
                "selection_probs": probs.tolist(),
                "fitness": [ar.fitness for ar in agent_responses]
            })

            # Build context from selected responses
            def context_provider(agent_id: int) -> str:
                selected_texts = [
                    f"Selected response: {s.response_text}"
                    for s in selected
                ]
                return "\n\n".join(selected_texts)

            # Generate new responses
            fs_task = Task(
                task_id=f"{task.task_id}_round{round_idx}",
                prompt=task.prompt,
                category=task.category
            )

            responses = system.generate_responses(fs_task, context_provider)

            # Evaluate quality
            if quality_evaluator:
                for r in responses:
                    r.quality_score = quality_evaluator(r.text, task)
            else:
                for r in responses:
                    r.quality_score = 0.75  # Fixed quality to isolate diversity mechanism

            round_history.append(deepcopy(responses))

        return {
            "final_responses": responses,
            "round_history": round_history,
            "selection_history": selection_history,
            "metrics": {
                "protocol": "fitness_sharing",
                "n_rounds": self.config.n_rounds,
                "share_radius": self.config.share_radius
            }
        }


def create_protocol(name: str, config: ProtocolConfig) -> Protocol:
    """Factory function to create protocols."""
    protocols = {
        "independent": IndependentProtocol,
        "debate": DebateProtocol,
        "debate_dds": DebateDDSProtocol,
        "diversity_prompt": DiversityPromptProtocol,
        "role_assignment": RoleAssignmentProtocol,
        "dds": DDSProtocol,
        "fitness_sharing": FitnessSharingProtocol,
        "novelty_search": NoveltySearchProtocol
    }

    if name not in protocols:
        raise ValueError(f"Unknown protocol: {name}. Available: {list(protocols.keys())}")

    return protocols[name](config)


if __name__ == "__main__":
    from .llm_agent import EXAMPLE_TASKS

    print("Testing Protocols with Mock Agents")
    print("=" * 50)

    # Create system with mock agents
    system = MultiAgentSystem(
        n_agents=5,
        agent_config_template=AgentConfig(agent_id=0, backend="mock")
    )

    task = EXAMPLE_TASKS[0]

    # Test each protocol
    protocols_to_test = [
        ("independent", ProtocolConfig(n_rounds=1, n_agents=5)),
        ("debate", DebateConfig(n_rounds=3, n_agents=5)),
        ("dds", DDSConfig(n_rounds=3, n_agents=5, alpha=1.0)),
    ]

    for name, config in protocols_to_test:
        print(f"\n--- {name.upper()} ---")
        protocol = create_protocol(name, config)
        result = protocol.run(system, task)
        print(f"Final responses: {len(result['final_responses'])}")
        print(f"Rounds: {len(result['round_history'])}")

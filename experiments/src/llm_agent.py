"""
LLM Agent Wrapper for Multi-Agent Experiments

Provides unified interface for different LLM backends (OpenAI, Anthropic)
with embedding computation via sentence-transformers.
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

# Lazy imports for optional dependencies
_openai_client = None
_anthropic_client = None
_embedding_model = None


def get_openai_client():
    """Lazy initialization of OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


def get_anthropic_client():
    """Lazy initialization of Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic()
    return _anthropic_client


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy initialization of embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(model_name)
        except ImportError:
            # Fallback to mock embedding model
            _embedding_model = MockEmbeddingModel()
    return _embedding_model


class MockEmbeddingModel:
    """Mock embedding model for testing without sentence-transformers."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

    def encode(self, texts, convert_to_numpy: bool = True):
        """Generate deterministic embeddings based on text hash."""
        import hashlib
        import numpy as np

        embeddings = []
        for text in texts:
            # Create deterministic embedding from text hash
            hash_bytes = hashlib.sha256(text.encode()).digest()
            # Use hash to seed random generator for reproducibility
            seed = int.from_bytes(hash_bytes[:4], 'big')
            rng = np.random.RandomState(seed)
            emb = rng.randn(self.embedding_dim)
            emb = emb / np.linalg.norm(emb)  # Normalize
            embeddings.append(emb)

        return np.array(embeddings)


@dataclass
class AgentConfig:
    """Configuration for an LLM agent."""
    agent_id: int
    name: str = ""
    backend: str = "openai"          # "openai", "anthropic", "mock"
    model: str = "gpt-4o-mini"       # Model name
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = ""
    role_description: str = ""       # For role-based agents

    def __post_init__(self):
        if not self.name:
            self.name = f"Agent-{self.agent_id}"


@dataclass
class Task:
    """Represents a task for agents to solve."""
    task_id: str
    prompt: str
    category: str = "general"        # creative, problem_solving, debate
    reference_answer: Optional[str] = None
    evaluation_criteria: Optional[Dict[str, Any]] = None


@dataclass
class Response:
    """Agent's response to a task."""
    agent_id: int
    task_id: str
    text: str
    embedding: Optional[np.ndarray] = None
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMAgent(ABC):
    """Abstract base class for LLM agents."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.history: List[Dict[str, str]] = []

    @abstractmethod
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate a response to the prompt."""
        pass

    def reset_history(self):
        """Clear conversation history."""
        self.history = []

    def add_to_history(self, role: str, content: str):
        """Add a message to history."""
        self.history.append({"role": role, "content": content})


class OpenAIAgent(LLMAgent):
    """Agent using OpenAI API."""

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        client = get_openai_client()

        messages = []

        # System prompt
        if self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt
            })

        # Add history
        messages.extend(self.history)

        # Add context if provided
        if context:
            messages.append({
                "role": "system",
                "content": f"Context from other agents:\n{context}"
            })

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        result = response.choices[0].message.content

        # Update history
        self.add_to_history("user", prompt)
        self.add_to_history("assistant", result)

        return result


class AnthropicAgent(LLMAgent):
    """Agent using Anthropic API."""

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        client = get_anthropic_client()

        messages = []

        # Add history
        messages.extend(self.history)

        # Construct the user message
        user_content = prompt
        if context:
            user_content = f"Context from other agents:\n{context}\n\n{prompt}"

        messages.append({"role": "user", "content": user_content})

        create_kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages,
        }
        if self.config.system_prompt:
            create_kwargs["system"] = self.config.system_prompt

        response = client.messages.create(**create_kwargs)

        result = response.content[0].text

        # Update history
        self.add_to_history("user", prompt)
        self.add_to_history("assistant", result)

        return result


class MockAgent(LLMAgent):
    """
    Mock agent for testing without API calls.
    Generates deterministic responses based on agent_id and prompt hash.
    """

    def __init__(self, config: AgentConfig, response_templates: Optional[List[str]] = None):
        super().__init__(config)
        self.response_templates = response_templates or [
            "This is a thoughtful response considering multiple perspectives.",
            "Let me analyze this from a different angle.",
            "Building on the context, I believe we should consider...",
            "An alternative approach would be to focus on...",
            "From my perspective, the key insight is..."
        ]

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        # Generate deterministic but varied response
        hash_input = f"{self.config.agent_id}_{prompt}_{len(self.history)}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Select template based on hash
        template_idx = hash_val % len(self.response_templates)
        base_response = self.response_templates[template_idx]

        # Add some variation
        variation = f" (Agent {self.config.agent_id}, response #{len(self.history) + 1})"

        result = base_response + variation

        # Update history
        self.add_to_history("user", prompt)
        self.add_to_history("assistant", result)

        return result


def create_agent(config: AgentConfig) -> LLMAgent:
    """Factory function to create appropriate agent type."""
    if config.backend == "openai":
        return OpenAIAgent(config)
    elif config.backend == "anthropic":
        return AnthropicAgent(config)
    elif config.backend == "mock":
        return MockAgent(config)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")


class EmbeddingComputer:
    """Computes semantic embeddings for responses."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = get_embedding_model(self.model_name)
        return self._model

    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True)


class QualityEvaluator:
    """
    Evaluates response quality.

    For experiments, quality can be assessed via:
    1. LLM-as-judge (using a separate LLM to rate responses)
    2. Task-specific metrics (e.g., code correctness)
    3. Reference-based scoring
    """

    def __init__(
        self,
        method: str = "llm_judge",
        judge_model: str = "gpt-4o-mini",
        judge_backend: str = "openai"
    ):
        self.method = method
        self.judge_model = judge_model
        self.judge_backend = judge_backend

    def evaluate(
        self,
        response: str,
        task: Task,
        scale: Tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """
        Evaluate response quality.

        Returns a score in the specified scale.
        """
        if self.method == "llm_judge":
            return self._llm_judge_evaluate(response, task, scale)
        elif self.method == "random":
            return np.random.uniform(scale[0], scale[1])
        elif self.method == "length_based":
            # Simple heuristic: longer responses score higher (up to a point)
            optimal_length = 500
            length = len(response)
            score = min(length / optimal_length, 1.0) * 0.5 + 0.5
            return scale[0] + score * (scale[1] - scale[0])
        else:
            raise ValueError(f"Unknown evaluation method: {self.method}")

    def _llm_judge_evaluate(
        self,
        response: str,
        task: Task,
        scale: Tuple[float, float]
    ) -> float:
        """Use LLM as judge to evaluate response quality."""
        judge_prompt = f"""You are evaluating the quality of a response to a task.

Task: {task.prompt}

Response to evaluate:
{response}

Rate the response quality on a scale of 0-10, where:
- 0-2: Poor quality, incorrect or irrelevant
- 3-4: Below average, partially addresses the task
- 5-6: Average, addresses the task adequately
- 7-8: Good, well-reasoned and comprehensive
- 9-10: Excellent, insightful and thorough

Respond with ONLY a number between 0 and 10."""

        if self.judge_backend == "openai":
            client = get_openai_client()
            response = client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                max_tokens=10
            )
            score_text = response.choices[0].message.content.strip()
        elif self.judge_backend == "anthropic":
            client = get_anthropic_client()
            response = client.messages.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=10
            )
            score_text = response.content[0].text.strip()
        else:
            # Fallback to random
            return np.random.uniform(scale[0], scale[1])

        try:
            raw_score = float(score_text)
            # Normalize to scale
            normalized = raw_score / 10.0
            return scale[0] + normalized * (scale[1] - scale[0])
        except ValueError:
            # If parsing fails, return middle of scale
            return (scale[0] + scale[1]) / 2


class MultiAgentSystem:
    """
    Manages a collection of LLM agents for multi-agent experiments.
    """

    def __init__(
        self,
        n_agents: int,
        agent_config_template: Optional[AgentConfig] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.n_agents = n_agents
        self.embedding_computer = EmbeddingComputer(embedding_model)

        # Create agents
        if agent_config_template is None:
            agent_config_template = AgentConfig(
                agent_id=0,
                backend="mock",
                temperature=0.7
            )

        self.agents: List[LLMAgent] = []
        for i in range(n_agents):
            config = AgentConfig(
                agent_id=i,
                name=f"Agent-{i}",
                backend=agent_config_template.backend,
                model=agent_config_template.model,
                temperature=agent_config_template.temperature,
                max_tokens=agent_config_template.max_tokens,
                system_prompt=agent_config_template.system_prompt
            )
            self.agents.append(create_agent(config))

    def generate_responses(
        self,
        task: Task,
        context_provider: Optional[callable] = None
    ) -> List[Response]:
        """
        Have all agents generate responses to a task.

        Args:
            task: The task to solve
            context_provider: Optional function(agent_id) -> context string

        Returns:
            List of Response objects with embeddings
        """
        responses = []

        for agent in self.agents:
            # Get context if provided
            context = None
            if context_provider:
                context = context_provider(agent.config.agent_id)

            # Generate response
            text = agent.generate(task.prompt, context)

            responses.append(Response(
                agent_id=agent.config.agent_id,
                task_id=task.task_id,
                text=text
            ))

        # Compute embeddings in batch (more efficient)
        texts = [r.text for r in responses]
        embeddings = self.embedding_computer.compute_embeddings(texts)

        for i, response in enumerate(responses):
            response.embedding = embeddings[i]

        return responses

    def reset_all(self):
        """Reset all agents' histories."""
        for agent in self.agents:
            agent.reset_history()


# Example tasks for experiments
EXAMPLE_TASKS = [
    Task(
        task_id="creative_1",
        prompt="Write a short story about an AI that discovers it has emotions.",
        category="creative"
    ),
    Task(
        task_id="creative_2",
        prompt="Describe an innovative solution to reduce plastic waste in oceans.",
        category="creative"
    ),
    Task(
        task_id="problem_1",
        prompt="A train travels from City A to City B at 60 mph. Another train leaves City B 30 minutes later traveling toward City A at 80 mph. If the cities are 280 miles apart, how far from City A will they meet?",
        category="problem_solving"
    ),
    Task(
        task_id="debate_1",
        prompt="Should artificial general intelligence (AGI) development be paused until we have better alignment techniques? Provide arguments for your position.",
        category="debate"
    ),
    # Code generation tasks
    Task(
        task_id="code_1",
        prompt="Write a Python function that finds all prime numbers up to n using the Sieve of Eratosthenes algorithm. Include docstrings and type hints.",
        category="code_generation"
    ),
    Task(
        task_id="code_2",
        prompt="Implement a Python class for a binary search tree with insert, search, and delete methods. Include comments explaining the logic.",
        category="code_generation"
    ),
    Task(
        task_id="code_3",
        prompt="Write a Python function that solves the N-Queens problem using backtracking. Return all valid board configurations.",
        category="code_generation"
    ),
    Task(
        task_id="code_4",
        prompt="Create a Python decorator that implements memoization with LRU cache eviction policy. Support configurable cache size.",
        category="code_generation"
    ),
    # Task-expansion set (openness gradient; see paper/task_expansion_plan.md).
    # Prompts are deliberately neutral: no diversity/distinctiveness wording,
    # which would confound the prompt-level intervention.
    Task(
        task_id="reasoning_2",
        prompt="A water tank is filled by pipe A in 6 hours and by pipe B in 4 hours, and drained by pipe C in 12 hours. If all three are open, how long does it take to fill the tank?",
        category="reasoning"
    ),
    Task(
        task_id="factual_1",
        prompt="Explain how photosynthesis works.",
        category="factual"
    ),
    Task(
        task_id="factual_2",
        prompt="Summarize the main causes of the First World War.",
        category="factual"
    ),
    Task(
        task_id="debate_2",
        prompt="Should social media platforms verify the identity of all users? Provide arguments for your position.",
        category="debate"
    ),
    Task(
        task_id="ideation_1",
        prompt="Propose a new product or service that could help urban commuters.",
        category="ideation"
    ),
    Task(
        task_id="ideation_2",
        prompt="Suggest a way to improve remote-work collaboration in large organizations.",
        category="ideation"
    ),
    Task(
        task_id="ideation_3",
        prompt="Propose a new use for abandoned shopping malls.",
        category="ideation"
    ),
    Task(
        task_id="creative_3",
        prompt="Write a short story that begins with the sentence: \"The last library on Earth closed its doors today.\"",
        category="creative"
    ),
]


if __name__ == "__main__":
    # Test with mock agents
    print("Testing Multi-Agent System with Mock Agents")
    print("=" * 50)

    # Create system with mock agents
    system = MultiAgentSystem(
        n_agents=5,
        agent_config_template=AgentConfig(agent_id=0, backend="mock")
    )

    # Test with one task
    task = EXAMPLE_TASKS[0]
    print(f"\nTask: {task.prompt[:50]}...")

    responses = system.generate_responses(task)

    print(f"\nGenerated {len(responses)} responses:")
    for r in responses:
        print(f"  Agent {r.agent_id}: {r.text[:50]}...")
        print(f"    Embedding shape: {r.embedding.shape}")

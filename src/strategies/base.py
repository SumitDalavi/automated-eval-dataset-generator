"""Base interfaces for dataset generation strategies."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class EvalSample:
    """A single evaluation sample for an LLM task."""
    id: str
    input: str
    expected_output: str
    difficulty: str  # easy | medium | hard | adversarial
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class GenerationStrategy(ABC):
    """Base class for all dataset generation strategies."""

    @abstractmethod
    def generate(self, topic: str, n_samples: int, **kwargs) -> List[EvalSample]:
        """Generate n_samples evaluation examples for the given topic."""
        ...

    def name(self) -> str:
        return self.__class__.__name__

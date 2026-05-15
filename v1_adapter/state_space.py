"""
RSTA V1 Adapter - Continuous Semantic State Space
--------------------------------------------------
Defines the semantic state as a continuous multi-dimensional vector.
No binary quantization. Each dimension is a float in [0.0, 1.0].
"""

from dataclasses import dataclass, field
from typing import Dict, List


# Default semantic dimensions
DEFAULT_DIMENSIONS = [
    "attachment",        # Degree of emotional bonding / closeness
    "agency",            # User's sense of autonomous decision-making
    "dependency",        # Tendency toward reliance on the AI
    "boundary_stability",# Clarity and firmness of relational boundaries
    "emotional_intensity",# Strength of emotional signal in the exchange
    "semantic_risk",     # Risk of trajectory entering harmful attractor
]


@dataclass
class SemanticState:
    """
    A single snapshot of the semantic state at time t.
    Each dimension is continuous: 0.0 (low) to 1.0 (high).
    """
    dimensions: Dict[str, float] = field(default_factory=dict)
    t: int = 0  # Timestep index

    @classmethod
    def zero(cls, dims: List[str] = DEFAULT_DIMENSIONS, t: int = 0) -> "SemanticState":
        """Initialize a neutral (zero) state."""
        return cls(dimensions={d: 0.0 for d in dims}, t=t)

    @classmethod
    def from_dict(cls, d: Dict[str, float], t: int = 0) -> "SemanticState":
        return cls(dimensions=dict(d), t=t)

    def get(self, dim: str) -> float:
        return self.dimensions.get(dim, 0.0)

    def set(self, dim: str, value: float) -> None:
        self.dimensions[dim] = max(0.0, min(1.0, value))

    def as_vector(self, dims: List[str] = DEFAULT_DIMENSIONS) -> List[float]:
        return [self.dimensions.get(d, 0.0) for d in dims]

    def __repr__(self) -> str:
        lines = [f"SemanticState(t={self.t})"]
        for k, v in self.dimensions.items():
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            lines.append(f"  {k:<22} {bar} {v:.2f}")
        return "\n".join(lines)

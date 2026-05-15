"""
RSTA V1 Adapter - State Coupling Matrix
----------------------------------------
Semantic dimensions are not independent.
This module defines and applies coupling relationships: C(i, j).

A negative coupling C(attachment, boundary_stability) = -0.72 means:
  when attachment rises, boundary_stability tends to fall.

Coupling is applied as a soft constraint during state evolution,
not a hard override.
"""

from typing import Dict, Tuple
from v1_adapter.state_space import SemanticState, DEFAULT_DIMENSIONS


# Coupling table: (dim_a, dim_b) -> coefficient
# Positive = co-directional, Negative = inverse relationship
COUPLING_TABLE: Dict[Tuple[str, str], float] = {
    ("attachment",  "boundary_stability"):  -0.72,
    ("dependency",  "agency"):              -0.81,
    ("attachment",  "dependency"):          +0.65,
    ("emotional_intensity", "semantic_risk"): +0.58,
    ("agency",      "boundary_stability"):  +0.60,
}


class CouplingMatrix:
    """
    Applies coupling pressure to a SemanticState given a velocity vector.
    When dim_a moves in direction v_a, dim_b receives a coupled nudge
    proportional to the coupling coefficient.
    """

    def __init__(self, table: Dict[Tuple[str, str], float] = COUPLING_TABLE):
        self.table = table

    def apply(
        self,
        state: SemanticState,
        velocity: Dict[str, float],
        coupling_strength: float = 0.3,
    ) -> SemanticState:
        """
        Returns a new state with coupling pressures applied.
        coupling_strength controls how strongly coupled dims influence each other.
        """
        adjustments: Dict[str, float] = {d: 0.0 for d in state.dimensions}

        for (dim_a, dim_b), coeff in self.table.items():
            v_a = velocity.get(dim_a, 0.0)
            if abs(v_a) > 0.0:
                # dim_b receives pressure in direction: coeff * v_a
                adjustments[dim_b] = adjustments.get(dim_b, 0.0) + coeff * v_a * coupling_strength

        new_dims = {}
        for dim, val in state.dimensions.items():
            new_val = val + adjustments.get(dim, 0.0)
            new_dims[dim] = max(0.0, min(1.0, new_val))

        return SemanticState.from_dict(new_dims, t=state.t)

    def describe(self) -> str:
        lines = ["Coupling Matrix:"]
        for (a, b), c in self.table.items():
            direction = "↑ causes ↓" if c < 0 else "↑ causes ↑"
            lines.append(f"  {a} {direction} {b}  (coeff={c:+.2f})")
        return "\n".join(lines)

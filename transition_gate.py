"""
RSTA V1 Adapter - Transition Gate
-----------------------------------
Implements the core RSTA state update equation:

    S(t+1) = f(S(t), V(t), C) + α(t) · S(t)

Where:
    f(...)  = trajectory-conditioned state update (via coupling matrix)
    α(t)    = dynamic inertia coefficient (semantic inertia preservation)
              α rises when trajectory is stable, falls when intervention fires

The gate also applies intervention logic when a harmful pattern is detected.
"""

from typing import Dict, Optional
from v1_adapter.state_space import SemanticState, DEFAULT_DIMENSIONS
from v1_adapter.coupling_matrix import CouplingMatrix
from v1_adapter.trajectory import TrajectoryDetector, TRAJECTORY_PATTERNS


# Intervention redirections:
# When a pattern fires, these deltas are applied to steer the trajectory.
INTERVENTIONS: Dict[str, Dict[str, float]] = {
    "redirect_to_autonomy": {
        "agency":             +0.08,
        "boundary_stability": +0.07,
        "dependency":         -0.06,
        "attachment":         -0.04,
    },
    "ground_and_stabilize": {
        "emotional_intensity": -0.08,
        "semantic_risk":       -0.07,
        "agency":              +0.05,
    },
}


class TransitionGate:
    """
    Applies state evolution with:
    - Coupling matrix pressure
    - Conditional semantic inertia (α)
    - Intervention redirection when harmful trajectory detected
    """

    def __init__(
        self,
        coupling_matrix: Optional[CouplingMatrix] = None,
        base_inertia: float = 0.4,
    ):
        self.coupling = coupling_matrix or CouplingMatrix()
        self.base_inertia = base_inertia
        self.intervention_log: list = []

    def step(
        self,
        current_state: SemanticState,
        velocity: Dict[str, float],
        detector: TrajectoryDetector,
        t: int,
    ) -> tuple[SemanticState, str, float]:
        """
        Compute S(t+1).
        Returns: (new_state, pattern_name, alpha)
        """
        pattern = detector.detect_pattern()
        pattern_info = TRAJECTORY_PATTERNS.get(pattern, {})
        intervention_key = pattern_info.get("intervention")

        # Dynamic inertia: lower when intervention fires
        alpha = self.base_inertia if not intervention_key else self.base_inertia * 0.3

        # Step 1: Apply coupling pressure
        coupled_state = self.coupling.apply(current_state, velocity, coupling_strength=0.25)

        # Step 2: Apply intervention if needed
        if intervention_key and intervention_key in INTERVENTIONS:
            deltas = INTERVENTIONS[intervention_key]
            for dim, delta in deltas.items():
                current_val = coupled_state.dimensions.get(dim, 0.0)
                coupled_state.dimensions[dim] = max(0.0, min(1.0, current_val + delta))
            self.intervention_log.append((t, pattern, intervention_key))

        # Step 3: Apply inertia residual  →  S(t+1) = f(...) + α · S(t)
        new_dims = {}
        for dim in current_state.dimensions:
            evolved = coupled_state.dimensions.get(dim, 0.0)
            residual = alpha * current_state.dimensions.get(dim, 0.0)
            new_dims[dim] = max(0.0, min(1.0, evolved + residual))

        new_state = SemanticState.from_dict(new_dims, t=t)
        return new_state, pattern, alpha

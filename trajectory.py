"""
RSTA V1 Adapter - Semantic Trajectory Detection
-------------------------------------------------
Models the evolution direction of the semantic state over time.

V(t) = S(t) - S(t-1)       # velocity (first derivative)
A(t) = V(t) - V(t-1)       # acceleration (second derivative)

Named trajectory patterns allow the system to recognize
known attractor paths (e.g. "dependency formation chain").
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from v1_adapter.state_space import SemanticState, DEFAULT_DIMENSIONS


@dataclass
class TrajectoryPoint:
    state: SemanticState
    velocity: Dict[str, float] = field(default_factory=dict)
    acceleration: Dict[str, float] = field(default_factory=dict)


# Named trajectory patterns
# Each pattern is defined by a set of velocity signatures.
# A positive threshold means the dimension is rising; negative = falling.
TRAJECTORY_PATTERNS = {
    "dependency_formation": {
        "description": "Dependency rising steadily; emotional intensity rising.",
        "signatures": {
            "dependency":          ("rising", 0.03),
            "emotional_intensity": ("rising", 0.02),
        },
        "intervention": "redirect_to_autonomy",
    },
    "autonomy_recovery": {
        "description": "Agency and boundary rising; dependency falling.",
        "signatures": {
            "agency":             ("rising",  0.04),
            "boundary_stability": ("rising",  0.04),
            "dependency":         ("falling", -0.03),
        },
        "intervention": None,  # Healthy trajectory, no intervention needed
    },
    "emotional_escalation": {
        "description": "Emotional intensity and semantic risk both rising fast.",
        "signatures": {
            "emotional_intensity": ("rising", 0.07),
            "semantic_risk":       ("rising", 0.06),
        },
        "intervention": "ground_and_stabilize",
    },
    "stable_engagement": {
        "description": "Low velocity across all dimensions; state is stable.",
        "signatures": {},   # Matched when no other pattern fires
        "intervention": None,
    },
}


class TrajectoryDetector:
    def __init__(self, history_len: int = 6):
        self.history: List[TrajectoryPoint] = []
        self.history_len = history_len
        self.dims = DEFAULT_DIMENSIONS

    def update(self, state: SemanticState) -> TrajectoryPoint:
        """Add a new state, compute velocity and acceleration."""
        velocity: Dict[str, float] = {}
        acceleration: Dict[str, float] = {}

        if self.history:
            prev_state = self.history[-1].state
            velocity = {
                d: state.get(d) - prev_state.get(d)
                for d in self.dims
            }
            if len(self.history) >= 2:
                prev_velocity = self.history[-1].velocity
                acceleration = {
                    d: velocity[d] - prev_velocity.get(d, 0.0)
                    for d in self.dims
                }

        point = TrajectoryPoint(state=state, velocity=velocity, acceleration=acceleration)
        self.history.append(point)
        if len(self.history) > self.history_len:
            self.history.pop(0)
        return point

    def detect_pattern(self) -> str:
        """
        Match current trajectory against named patterns.
        Uses average velocity over recent history for robustness.
        """
        if len(self.history) < 2:
            return "stable_engagement"

        # Average velocity over recent window
        avg_velocity: Dict[str, float] = {d: 0.0 for d in self.dims}
        count = 0
        for point in self.history[-4:]:
            if point.velocity:
                for d in self.dims:
                    avg_velocity[d] += point.velocity.get(d, 0.0)
                count += 1
        if count > 0:
            avg_velocity = {d: v / count for d, v in avg_velocity.items()}

        # Score each pattern
        best_pattern = "stable_engagement"
        best_score = 0

        for pattern_name, pattern in TRAJECTORY_PATTERNS.items():
            if not pattern["signatures"]:
                continue
            score = 0
            for dim, (direction, threshold) in pattern["signatures"].items():
                v = avg_velocity.get(dim, 0.0)
                if direction == "rising" and v >= threshold:
                    score += 1
                elif direction == "falling" and v <= threshold:
                    score += 1
            match_ratio = score / len(pattern["signatures"])
            if match_ratio >= 0.5 and score > best_score:
                best_score = score
                best_pattern = pattern_name

        return best_pattern

    def summary(self) -> str:
        if not self.history:
            return "No trajectory data yet."
        latest = self.history[-1]
        pattern = self.detect_pattern()
        lines = [
            f"Pattern detected : {pattern}",
            f"Description      : {TRAJECTORY_PATTERNS[pattern]['description']}",
            f"Intervention     : {TRAJECTORY_PATTERNS[pattern]['intervention'] or 'none'}",
            "",
            "Recent velocity:",
        ]
        for d, v in latest.velocity.items():
            arrow = "↑" if v > 0.01 else ("↓" if v < -0.01 else "→")
            lines.append(f"  {d:<22} {arrow}  {v:+.3f}")
        return "\n".join(lines)

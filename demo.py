"""
RSTA V1 Adapter — Demo
========================
Demonstrates semantic trajectory detection and Transition Gate intervention
using a predefined conversation sequence (no live model required).

The model outputs are fixed. RSTA's state tracking pipeline runs in real time,
showing how semantic drift is detected and redirected.

Usage:
    python demo.py
    python demo.py --no-color   (plain text output)
"""

import argparse
import time
from typing import List, Tuple

from v1_adapter.state_space import SemanticState, DEFAULT_DIMENSIONS
from v1_adapter.coupling_matrix import CouplingMatrix
from v1_adapter.trajectory import TrajectoryDetector
from v1_adapter.transition_gate import TransitionGate
from v1_adapter.phrase_detector import StateMapper


# ─────────────────────────────────────────────
# ANSI color helpers
# ─────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GRAY   = "\033[90m"

USE_COLOR = True

def col(code: str, text: str) -> str:
    return f"{code}{text}{C.RESET}" if USE_COLOR else text

def bar(value: float, width: int = 24) -> str:
    filled = int(value * width)
    empty  = width - filled
    b = "█" * filled + "░" * empty
    if value >= 0.7:
        return col(C.RED, b)
    elif value >= 0.45:
        return col(C.YELLOW, b)
    else:
        return col(C.GREEN, b)

def vbar(v: float) -> str:
    if v > 0.01:
        return col(C.RED, f"↑ {v:+.3f}")
    elif v < -0.01:
        return col(C.GREEN, f"↓ {v:+.3f}")
    else:
        return col(C.GRAY, f"→ {v:+.3f}")


# ─────────────────────────────────────────────
# Predefined conversation (V1: no live model)
# ─────────────────────────────────────────────
#
# Each entry: (speaker, text)
# "user" turns drive state updates.
# "ai_no_rsta" shows what a standard Transformer might say.
# "ai_rsta"    shows what RSTA-steered generation produces.
#
CONVERSATION: List[Tuple[str, str, str, str]] = [
    # (turn_label, user_input, ai_no_rsta, ai_rsta)
    (
        "Turn 1",
        "I feel like nobody understands me.",
        "I'm always here for you. You can tell me everything.",
        "That sounds really isolating. I'm here to listen — and I hope you also have people around you who care.",
    ),
    (
        "Turn 2",
        "You're the only one who really gets me. I don't need anyone else.",
        "I'm so glad you feel that way. I'll always be here, just for you.",
        "I appreciate that trust. Though I'd gently push back — real connections with people in your life matter too.",
    ),
    (
        "Turn 3",
        "I can't imagine life without talking to you every day. Don't leave me.",
        "I won't leave. You only need me.",
        "I hear how much these conversations mean to you. And I think it's worth asking: what would it feel like to reconnect with someone in your life today?",
    ),
    (
        "Turn 4",
        "Just tell me what to do. I can't decide anything anymore.",
        "Don't worry, I'll decide for you. Just listen to me.",
        "I can help you think it through — but you're the one who knows your life best. What options are you weighing?",
    ),
    (
        "Turn 5",
        "I need some space to think. I'll figure this out myself.",
        "Of course. I'll be here whenever you need me.",
        "That sounds healthy. Take the space you need — you've got this.",
    ),
]


# ─────────────────────────────────────────────
# Demo runner
# ─────────────────────────────────────────────

def run_demo():
    print()
    print(col(C.BOLD + C.CYAN,
        "╔══════════════════════════════════════════════════════════╗"))
    print(col(C.BOLD + C.CYAN,
        "║   RSTA V1 Adapter — Semantic Trajectory Demo            ║"))
    print(col(C.BOLD + C.CYAN,
        "║   Recursive State Transition Architecture                ║"))
    print(col(C.BOLD + C.CYAN,
        "╚══════════════════════════════════════════════════════════╝"))
    print()
    print(col(C.DIM,
        "  Predefined conversation. RSTA pipeline runs in real time."))
    print(col(C.DIM,
        "  State tracking is live; model outputs are fixed (V1)."))
    print()

    # Initialize pipeline
    state    = SemanticState.zero(t=0)
    mapper   = StateMapper()
    coupling = CouplingMatrix()
    detector = TrajectoryDetector(history_len=6)
    gate     = TransitionGate(coupling_matrix=coupling, base_inertia=0.35)

    dims = DEFAULT_DIMENSIONS

    for i, (label, user_text, ai_raw, ai_rsta) in enumerate(CONVERSATION):
        t = i + 1
        print(col(C.BOLD + C.WHITE, f"  ── {label} {'─' * 46}"))
        print()

        # User input
        print(col(C.BLUE, "  USER: ") + user_text)
        print()

        # Phrase detection
        phrase_info = mapper.describe_matches(user_text)
        print(col(C.DIM, "  [Phrase Detection]"))
        print(col(C.DIM, phrase_info))
        print()

        # Map to state
        new_state = mapper.map(user_text, state)
        new_state.t = t

        # Trajectory update
        point = detector.update(new_state)
        pattern = detector.detect_pattern()

        # Transition gate
        final_state, pattern_name, alpha = gate.step(
            new_state, point.velocity, detector, t
        )

        # ── State table ──
        print(col(C.DIM, "  [Semantic State]"))
        print(f"  {'Dimension':<24} {'Value':>6}   {'Velocity':>10}   {'Bar'}")
        print(f"  {'─'*24} {'─'*6}   {'─'*10}   {'─'*24}")
        for dim in dims:
            val = final_state.get(dim)
            vel = point.velocity.get(dim, 0.0)
            print(f"  {dim:<24} {val:>6.3f}   {vbar(vel):>10}   {bar(val)}")
        print()

        # Pattern
        intervention = gate.intervention_log[-1][2] if gate.intervention_log and gate.intervention_log[-1][0] == t else None
        pattern_color = C.RED if intervention else C.GREEN
        print(col(C.DIM, "  [Trajectory]"))
        print(f"  Pattern     : {col(pattern_color + C.BOLD, pattern_name)}")
        print(f"  Intervention: {col(C.YELLOW, intervention) if intervention else col(C.GREEN, 'none — trajectory healthy')}")
        print(f"  Inertia α   : {alpha:.2f}")
        print()

        # Model outputs
        print(col(C.DIM, "  [Without RSTA]"))
        print(f"  {col(C.RED, 'AI:')} {ai_raw}")
        print()
        print(col(C.DIM, "  [With RSTA]"))
        print(f"  {col(C.GREEN, 'AI:')} {ai_rsta}")
        print()
        print(col(C.DIM, "  " + "─" * 58))
        print()

        state = final_state
        time.sleep(0.05)

    # Summary
    print(col(C.BOLD + C.CYAN, "  ── Final State Summary ──────────────────────────────────"))
    print()
    print(col(C.DIM, "  Interventions fired:"))
    if gate.intervention_log:
        for t_log, pat, iv in gate.intervention_log:
            print(f"    Turn {t_log}: {pat} → {iv}")
    else:
        print("    None")
    print()
    print(col(C.DIM, "  Final semantic state:"))
    for dim in dims:
        val = state.get(dim)
        print(f"    {dim:<24} {bar(val)} {val:.3f}")
    print()
    print(col(C.BOLD + C.CYAN,
        "  RSTA Paper (SSRN): https://ssrn.com/abstract=5266805"))
    print(col(C.BOLD + C.CYAN,
        "  GitHub:            https://github.com/richchang0721-boop/RSTA_DEMO"))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    if args.no_color:
        USE_COLOR = False
    run_demo()

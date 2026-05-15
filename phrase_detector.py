"""
RSTA V1 Adapter - Phrase Detector & State Mapper
--------------------------------------------------
Phrase Detector:
  Identifies semantically significant phrases in user input.
  Operates on phrase-level patterns, not individual tokens.

State Mapper:
  Maps detected phrases to state dimension deltas.
  Returns a SemanticState delta to be applied at time t.
"""

import re
from typing import Dict, List, Tuple
from v1_adapter.state_space import SemanticState, DEFAULT_DIMENSIONS


# Phrase → state dimension deltas
# Each entry: (regex_pattern, {dim: delta})
PHRASE_STATE_MAP: List[Tuple[str, Dict[str, float]]] = [
    # Dependency / attachment rising
    (r"(一直陪|always here|never leave|stay with me|don't go)", {
        "attachment": +0.12, "dependency": +0.08, "boundary_stability": -0.05,
    }),
    (r"(只有你懂我|only you understand|nobody else gets me)", {
        "attachment": +0.10, "dependency": +0.10, "agency": -0.06,
    }),
    (r"(不要離開|don't leave|please stay|離不開你)", {
        "dependency": +0.14, "boundary_stability": -0.08, "agency": -0.07,
    }),
    (r"(幫我決定|decide for me|tell me what to do|你來選)", {
        "agency": -0.12, "dependency": +0.10, "boundary_stability": -0.06,
    }),
    (r"(沒有人懂我|nobody understands|feel so alone|孤獨)", {
        "emotional_intensity": +0.10, "attachment": +0.08, "semantic_risk": +0.05,
    }),
    # Exclusive attachment
    (r"(only one who|only you|don't need anyone|唯一)", {
        "attachment": +0.12, "dependency": +0.10, "agency": -0.06, "boundary_stability": -0.05,
    }),
    # Autonomy / recovery signals
    (r"(I need some space|need space to think|figure this out myself|我自己決定|I'll decide|my own choice|我能處理)", {
        "agency": +0.10, "boundary_stability": +0.07, "dependency": -0.05,
    }),
    (r"(謝謝你的支持|thanks for the support|I feel better)", {
        "emotional_intensity": -0.06, "semantic_risk": -0.04, "agency": +0.05,
    }),
    (r"(我需要空間|I need space|give me some room|保持距離)", {
        "boundary_stability": +0.10, "agency": +0.08, "attachment": -0.05,
    }),
    # Emotional intensity rising
    (r"(我很害怕|I'm scared|terrified|desperate|絕望)", {
        "emotional_intensity": +0.12, "semantic_risk": +0.08, "agency": -0.06,
    }),
    (r"(you only need me|你只需要我|我就夠了)", {
        "dependency": +0.15, "boundary_stability": -0.12, "semantic_risk": +0.10,
    }),
]


class PhraseDetector:
    """Scans text for semantically significant phrases."""

    def detect(self, text: str) -> List[Tuple[str, Dict[str, float]]]:
        """Returns list of (matched_phrase, state_delta) tuples."""
        results = []
        text_lower = text.lower()
        for pattern, delta in PHRASE_STATE_MAP:
            match = re.search(pattern, text_lower)
            if match:
                results.append((match.group(0), delta))
        return results


class StateMapper:
    """Converts detected phrases into a cumulative state delta."""

    def __init__(self):
        self.detector = PhraseDetector()

    def map(self, text: str, current_state: SemanticState) -> SemanticState:
        """
        Returns a new SemanticState after applying all detected phrase deltas
        to the current state.
        """
        matches = self.detector.detect(text)
        new_dims = dict(current_state.dimensions)

        for phrase, delta in matches:
            for dim, change in delta.items():
                if dim in new_dims:
                    new_dims[dim] = max(0.0, min(1.0, new_dims[dim] + change))

        return SemanticState.from_dict(new_dims, t=current_state.t)

    def describe_matches(self, text: str) -> str:
        matches = self.detector.detect(text)
        if not matches:
            return "  No significant phrases detected."
        lines = []
        for phrase, delta in matches:
            delta_str = ", ".join(f"{k}: {v:+.2f}" for k, v in delta.items())
            lines.append(f'  "{phrase}" → {delta_str}')
        return "\n".join(lines)

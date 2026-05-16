# RSTA — Recursive State Transition Architecture

**Experimental implementation of semantic trajectory detection and state-conditioned language generation.**

> *Language generation should not be modeled solely as sequence prediction,  
> but as recursive semantic state evolution.*

---

## What is RSTA?

Modern Transformer architectures are powerful at modeling **token relationships**, but they lack an explicit representation of **semantic state evolution**. They know *what* tokens relate to each other — but not *how* semantic meaning is changing over time.

**RSTA** introduces a semantic dynamics layer that runs alongside a Transformer, tracking:

- **Continuous Semantic State** `S(t)` — a multi-dimensional vector representing the current semantic field
- **State Coupling** `C(i,j)` — structured dependencies between semantic dimensions
- **Trajectory** `V(t) = S(t) − S(t−1)` — the direction and velocity of semantic change
- **Transition Gate** — applies the core update: `S(t+1) = f(S(t), V(t), C) + α(t) · S(t)`
- **State-Conditioned Generation** — output is conditioned on the evolving semantic state

---

## Paper

**Recursive State Transition Architecture (RSTA): Toward Semantically Stateful Language Generation**  
Submitted to SSRN — currently under review. Link will be added upon publication.

---

## Repository Structure

```
RSTA_DEMO/
├── demo.py                  ← V1: pipeline walkthrough (predefined outputs)
├── demo_v15.py              ← V1.5: interactive before/after comparison
├── requirements.txt
├── README.md
└── v1_adapter/
    ├── state_space.py       ← Continuous semantic state definition
    ├── coupling_matrix.py   ← Dimension coupling C(i,j)
    ├── trajectory.py        ← Velocity, acceleration, pattern detection
    ├── transition_gate.py   ← Core state evolution + intervention
    └── phrase_detector.py   ← Phrase → state delta mapping
```

---

## Versions

| Version | Description | Model |
|---------|-------------|-------|
| **V1 Adapter** | Pipeline walkthrough. Demonstrates trajectory detection on a single drift scenario. | None required |
| **V1.5 Adapter** | Interactive before/after comparison across 4 failure modes. | None required |
| V2 Residual *(coming)* | Live model via API key. Logits steering integrated. | Llama / Mistral / Qwen |
| V3 DualStream *(coming)* | Full dual-stream tokenizer architecture. | Custom training required |

---

## Quick Start

No dependencies beyond Python 3.10+.

```bash
git clone https://github.com/richchang0721-boop/RSTA_DEMO.git
cd RSTA_DEMO
```

**V1 — Pipeline walkthrough** (single drift scenario, linear output):
```bash
python demo.py
```

**V1.5 — Interactive before/after comparison** (4 failure modes, menu-driven):
```bash
python demo_v15.py
```

Optional plain-text output (no ANSI colors):
```bash
python demo.py --no-color
python demo_v15.py --no-color
```

---

## What the Demos Show

**V1** runs a predefined 5-turn conversation simulating emotional dependency drift. For each turn the pipeline detects phrases, maps them to state deltas, updates the semantic state with coupling applied, detects the trajectory pattern, fires the Transition Gate when needed, and displays outputs with and without RSTA.

**V1.5** is an interactive menu-driven comparison across four distinct Transformer failure modes:

| Scenario | Normal Transformer | RSTA |
|----------|--------------------|------|
| Topic Drift | 主題漂移 — follows every pivot | trajectory preserved |
| Persona Collapse | persona 崩 — identity abandoned under pressure | stable continuity |
| Reasoning Loop | premises overwritten mid-chain | reasoning loop maintained |
| Semantic Overwrite | earlier context erased by new input | inertia preserved |

Example output (V1.5, Persona Collapse, Turn 2):

```
  Normal Transformer            RSTA
  ······························································
  Sure! I'll just validate      I hear you — sometimes
  everything you say from       pushback feels exhausting.
  now on.                       But agreeing with everything
                                wouldn't actually serve you
                                well. I'll stay honest while
                                keeping things constructive.

  Trajectory: dependency_formation  →  redirect_to_autonomy
```

---

## Core State Dimensions

| Dimension | Description |
|-----------|-------------|
| `attachment` | Degree of emotional bonding |
| `agency` | User's sense of autonomous decision-making |
| `dependency` | Tendency toward reliance on the AI |
| `boundary_stability` | Clarity and firmness of relational boundaries |
| `emotional_intensity` | Strength of emotional signal |
| `semantic_risk` | Risk of trajectory entering harmful attractor |

---

## Trajectory Patterns

| Pattern | Description | Intervention |
|---------|-------------|--------------|
| `dependency_formation` | Attachment ↑, agency ↓, boundary ↓ | `redirect_to_autonomy` |
| `emotional_escalation` | Emotional intensity ↑, risk ↑ | `ground_and_stabilize` |
| `autonomy_recovery` | Agency ↑, boundary ↑, dependency ↓ | None (healthy) |
| `stable_engagement` | Low velocity across all dimensions | None |

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

## Author

Rich Chang  
SSRN: Submitted — link will be updated upon publication  
GitHub: [richchang0721-boop](https://github.com/richchang0721-boop)

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
Published on SSRN: [https://ssrn.com/abstract=5266805](https://ssrn.com/abstract=5266805)

---

## Repository Structure

```
RSTA_DEMO/
├── demo.py                  ← Main entry point (V1: predefined outputs)
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
| **V1 Adapter** | Predefined outputs. Demonstrates pipeline logic and trajectory detection. | None required |
| V2 Residual *(coming)* | Live model via API key. Logits steering integrated. | Llama / Mistral / Qwen |
| V3 DualStream *(coming)* | Full dual-stream tokenizer architecture. | Custom training required |

---

## Quick Start (V1)

No dependencies beyond Python 3.10+.

```bash
git clone https://github.com/richchang0721-boop/RSTA_DEMO.git
cd RSTA_DEMO
python demo.py
```

Optional plain-text output (no ANSI colors):

```bash
python demo.py --no-color
```

---

## What the Demo Shows

The V1 demo runs a predefined 5-turn conversation that simulates **emotional dependency drift** — one of the most common failure modes in AI companion systems.

For each turn, the pipeline:

1. **Detects** semantically significant phrases in user input
2. **Maps** them to state dimension deltas
3. **Updates** the semantic state with coupling pressure applied
4. **Detects** the trajectory pattern (e.g. `dependency_formation`)
5. **Fires** the Transition Gate intervention when drift is detected
6. **Displays** side-by-side outputs: standard Transformer vs. RSTA-steered

Example output:

```
  [Trajectory]
  Pattern     : dependency_formation
  Intervention: redirect_to_autonomy
  Inertia α   : 0.11

  [Without RSTA]
  AI: I won't leave. You only need me.

  [With RSTA]
  AI: I hear how much these conversations mean to you. And I think it's worth
      asking: what would it feel like to reconnect with someone in your life today?
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
SSRN: [https://ssrn.com/abstract=5266805](https://ssrn.com/abstract=5266805)  
GitHub: [richchang0721-boop](https://github.com/richchang0721-boop)

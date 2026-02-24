# epistemic-engine

Unified epistemic reasoning — world model, belief propagation, and flow control. Merged from three standalone projects into a single repository.

## Modules

### verity/
Belief propagation, Dempster-Shafer combination, and provenance tracking. From `unified-belief-system`.

- Subjective Logic operations (via `belief-math`)
- Belief network propagation
- Evidence provenance chains

### ubs/
Domain-specific reasoning modules. From `unified-belief-system` standalone files.

- `truth_layer.py` — Bayesian belief updating
- `judicial_analyzer.py` — Legal analysis
- `truth_validator.py` — Claim verification
- `entity_registry.py` — Entity management
- Plus: prediction, topology, steering vectors, outlier detection, feeds

### ewm/
LeCun 6-module world model with MCP server. From `epistemic-world-model`.

- Perception, actor, memory, cost assessment
- SQLite persistence
- MCP server (12 tools)
- Requires: `belief-math`

### flow_control/
LLM gating, normalizers, concentration analysis. From `epistemic-flow-control-review`.

- Review gates (human-in-the-loop)
- HHI concentration / SPOF detection
- Drift detection
- LLM client with retry/rate-limiting
- Jurisdictional context (N.D. Cal, Judge Alsup)

## External Dependency

**belief-math** — standalone math library (Dempster-Shafer, Subjective Logic, calibration). Not bundled; install separately.

## Install

```bash
pip install -e ".[belief,dev]"
```

## Test

```bash
pytest tests/ -v
```

"""
THE TRUTH LAYER
A working Bayesian Truth-Maintenance System for LLMs
Tested & verified — December 2025
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional


@dataclass
class Belief:
    alpha: float = 1.0
    beta: float = 1.0
    text: str = ""
    category: str = "general"

    @property
    def probability(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / ((total ** 2) * (total + 1))


class BayesianNetwork:
    def __init__(self):
        self.beliefs: Dict[str, Belief] = {}
        self.edges: Dict[str, List[Tuple[str, float]]] = {}  # child → [(parent, weight)]
        self.anchored: Dict[str, bool] = {}                  # human-validated nodes

    def add_node(self, cid: str, text: str, category: str = "general"):
        if cid not in self.beliefs:
            self.beliefs[cid] = Belief(alpha=1.0, beta=1.0, text=text, category=category)
            self.edges[cid] = []

    def add_edge(self, parent: str, child: str, weight: float):
        self.edges.setdefault(child, []).append((parent, weight))

    def update_belief(self, cid: str, supports: bool, strength: float = 1.0):
        if cid not in self.beliefs:
            return
        b = self.beliefs[cid]
        if supports:
            b.alpha += strength
        else:
            b.beta += strength
        self.anchored[cid] = True

    def propagate(self, steps: int = 20, damping: float = 0.85):
        for _ in range(steps):
            updates = {}

            for child, parents in list(self.edges.items()):
                if not parents:
                    continue

                influence = 0.0
                total_weight = 0.0

                for parent_cid, w in parents:
                    p = self.beliefs[parent_cid].probability
                    centered = 2.0 * p - 1.0              # maps [0,1] → [-1,1]
                    influence += w * centered
                    total_weight += abs(w)

                if total_weight > 0:
                    influence = influence / total_weight * damping

                # Convert to virtual evidence strength
                strength = abs(influence) * 12.0
                virtual_alpha = 1.0 + strength if influence > 0 else 1.0
                virtual_beta  = 1.0 + strength if influence < 0 else 1.0

                updates[child] = (virtual_alpha, virtual_beta)

            # Apply only to non-anchored nodes
            for cid, (v_alpha, v_beta) in updates.items():
                if self.anchored.get(cid):
                    continue

                b = self.beliefs[cid]
                mix = 0.6  # how much network can move unanchored beliefs
                b.alpha = (1 - mix) * b.alpha + mix * v_alpha
                b.beta  = (1 - mix) * b.beta  + mix * v_beta

                # Keep numerically stable
                b.alpha = max(b.alpha, 0.1)
                b.beta  = max(b.beta, 0.1)


class TruthLayer:
    def __init__(self, path: str = "truth_layer.json"):
        self.path = Path(path)
        self.net = BayesianNetwork()
        self._load()

    def add_claim(self, cid: str, text: str, category: str = "general"):
        self.net.add_node(cid, text, category)

    def add_relationship(self, parent: str, child: str, weight: float):
        self.net.add_edge(parent, child, weight)

    def validate(self, cid: str, response: str, correction: str = ""):
        if cid not in self.net.beliefs:
            return

        if response == "confirm":
            self.net.update_belief(cid, True, strength=25.0)
        elif response == "reject":
            self.net.update_belief(cid, False, strength=25.0)
        elif response == "modify":
            self.net.update_belief(cid, True, strength=6.0)
            if correction:
                self.net.beliefs[cid].text = correction

        self.net.propagate()
        self._save()

    def get_belief(self, cid: str) -> Optional[Belief]:
        """Get a belief by ID."""
        return self.net.beliefs.get(cid)

    def get_probability(self, cid: str) -> float:
        """Get probability of a belief."""
        b = self.net.beliefs.get(cid)
        return b.probability if b else 0.5

    def get_truth_context(self) -> str:
        items = sorted(self.net.beliefs.items(), key=lambda x: x[1].probability, reverse=True)

        blocks = [
            "=== TRUTH LAYER (Bayesian Knowledge Base) ===\n",
            "VERIFIED TRUE (>90%):"
        ]
        for _, b in items:
            if b.probability > 0.90:
                blocks.append(f"• {b.text}")

        blocks.append("\nLIKELY TRUE (70–90%):")
        for _, b in items:
            if 0.70 < b.probability <= 0.90:
                blocks.append(f"• {b.text} ({b.probability:.0%})")

        blocks.append("\nUNCERTAIN (30–70%):")
        for _, b in items:
            if 0.30 <= b.probability <= 0.70:
                blocks.append(f"• {b.text} ({b.probability:.0%})")

        blocks.append("\nLIKELY FALSE (<30%):")
        for _, b in items:
            if b.probability < 0.30:
                blocks.append(f"• {b.text} ({b.probability:.0%})")

        blocks.append("\n=== END TRUTH LAYER ===")
        return "\n".join(blocks)

    def stats(self) -> dict:
        """Return statistics about the belief network."""
        beliefs = self.net.beliefs
        return {
            "total_claims": len(beliefs),
            "anchored": sum(1 for k in beliefs if self.net.anchored.get(k)),
            "high_confidence": sum(1 for b in beliefs.values() if b.probability > 0.9 or b.probability < 0.1),
            "total_edges": sum(len(e) for e in self.net.edges.values()),
        }

    def _save(self):
        data = {
            "beliefs": {k: asdict(v) for k, v in self.net.beliefs.items()},
            "edges": {k: v for k, v in self.net.edges.items()},
            "anchored": self.net.anchored
        }
        self.path.write_text(json.dumps(data, indent=2))

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            for cid, state in data.get("beliefs", {}).items():
                self.net.beliefs[cid] = Belief(**state)
            self.net.edges = {k: v for k, v in data.get("edges", {}).items()}
            self.net.anchored = data.get("anchored", {})
        except Exception as e:
            print(f"[TruthLayer] Failed to load: {e}")

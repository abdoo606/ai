"""Load the HCP structural connectome and build the brain from it."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .core import Agent, Brain
from .regions import HomeostatAgent, make_agent, parse_label

DATA_DIR = Path(__file__).resolve().parent / "data"
MATRIX_FILE = "hcp_structural_connectome_82.csv"
LABELS_FILE = "hcp_structural_labels_82.txt"


def load_structural(data_dir: Path = DATA_DIR) -> Tuple[List[str], List[List[float]]]:
    """Return (labels, matrix) for the 82-region HCP structural connectome."""
    text = (data_dir / LABELS_FILE).read_text()
    labels = [x.strip() for x in text.replace(",", "\n").splitlines() if x.strip()]
    matrix: List[List[float]] = []
    for line in (data_dir / MATRIX_FILE).read_text().splitlines():
        if line.strip():
            matrix.append([float(x) for x in line.split(",")])
    if len(matrix) != len(labels) or any(len(r) != len(labels) for r in matrix):
        raise ValueError("connectome matrix and label count do not match")
    return labels, matrix


def build_hcp_brain(
    *,
    coupling: float = 2.0,
    delay: int = 2,
    keep_fraction: float = 0.2,
    plastic: bool = True,
    extras: bool = True,
    data_dir: Path = DATA_DIR,
    brain: Optional[Brain] = None,
) -> Brain:
    """Build one agent per region and wire them with the connectome.

    Weights are scaled so the strongest connection equals ``coupling``.
    ``keep_fraction`` keeps only the strongest share of each region's connections.
    The matrix is undirected, so both directions get the same weight.
    Extra agents (cerebellum, hypothalamus) are not in the connectome and are
    wired by hand.
    """
    labels, sc = load_structural(data_dir)
    brain = brain or Brain()
    for label in labels:
        brain.add_agent(make_agent(label))

    peak = max(max(row) for row in sc)
    for i, src in enumerate(labels):
        # Keep the strongest ``keep_fraction`` of each region's own connections,
        # so no region is cut off from the network entirely.
        row = sorted((v for v in sc[i] if v > 0), reverse=True)
        keep = max(1, round(keep_fraction * len(row))) if row else 0
        cutoff = row[keep - 1] if row else 0.0
        for j, dst in enumerate(labels):
            if i == j or sc[i][j] <= 0 or sc[i][j] < cutoff:
                continue
            w = coupling * sc[i][j] / peak
            brain.connect(src, dst, w, delay=delay, plastic=plastic)

    if extras:
        add_extras(brain, labels)
    return brain


def add_extras(brain: Brain, labels: Sequence[str]) -> None:
    """Add regions that are not in the connectome: cerebellum and hypothalamus."""
    brain.add_agent(Agent("cerebellum", role="cerebellum", module="cerebellum"))
    brain.add_agent(HomeostatAgent("hypothalamus", role="hypothalamus",
                                   module="homeostasis", energy_budget=500.0))
    motor = [l for l in labels if parse_label(l)[1] == "precentral"]
    for m in motor:
        # Motor cortex drives the cerebellum; the cerebellum corrects it back.
        brain.connect(m, "cerebellum", 2.0, delay=2)
        brain.connect("cerebellum", m, -0.3, delay=3)

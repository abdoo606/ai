"""Visualisations of the brain's activity over time.

Needs matplotlib (not required by the model itself):
    pip install matplotlib

* ``activity_heatmap``: every agent (rows, grouped by module) against time (columns).
* ``graph_snapshot``:   the network as a graph; node size and colour show activity
                        in a time window, and strong connections are drawn.
* ``activity_gif``:     the graph snapshot animated over the whole run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from .core import Brain

MODULE_COLORS: Dict[str, str] = {
    "visual": "#4e79a7",
    "auditory": "#76b7b2",
    "somatosensory": "#59a14f",
    "motor": "#e15759",
    "memory": "#b07aa1",
    "control": "#f28e2b",
    "striatum": "#edc948",
    "thalamus": "#9c755f",
    "association": "#bab0ac",
    "cerebellum": "#d37295",
    "brainstem": "#499894",
    "homeostasis": "#79706e",
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _ordered_agents(brain: Brain) -> List[str]:
    order = list(MODULE_COLORS)
    return sorted(brain.agents, key=lambda n: (order.index(brain.agents[n].module)
                                               if brain.agents[n].module in order else len(order), n))


def activity_heatmap(brain: Brain, path: Path, bin_ms: int = 5, title: str = "") -> Path:
    """Save a heatmap of spikes per agent per ``bin_ms`` time bin."""
    plt = _plt()
    names = _ordered_agents(brain)
    n_bins = max(1, (brain.tick + bin_ms - 1) // bin_ms)
    grid = [[0] * n_bins for _ in names]
    for r, name in enumerate(names):
        for s in brain.agents[name].spikes:
            grid[r][s // bin_ms] += 1

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(grid, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel(f"time (bins of {bin_ms} ms)")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=5)
    # Colour the module labels
    for tick_label, name in zip(ax.get_yticklabels(), names):
        tick_label.set_color(MODULE_COLORS.get(brain.agents[name].module, "black"))
    fig.colorbar(im, ax=ax, label="spikes per bin")
    ax.set_title(title or "Activity of every region over time")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _layout(brain: Brain, names: Sequence[str]) -> Dict[str, tuple]:
    """Place agents on a circle, grouped by module so each module forms an arc."""
    import math
    modules: Dict[str, List[str]] = {}
    for n in names:
        modules.setdefault(brain.agents[n].module, []).append(n)
    pos: Dict[str, tuple] = {}
    total = len(names)
    k = 0
    for module in sorted(modules):
        for n in sorted(modules[module]):
            angle = 2 * math.pi * k / total
            pos[n] = (math.cos(angle), math.sin(angle))
            k += 1
    return pos


def graph_snapshot(brain: Brain, t0: int, t1: int, path: Path, *,
                   edge_quantile: float = 0.9, title: str = "") -> Path:
    """Draw the network with activity in the window [t0, t1)."""
    plt = _plt()
    names = _ordered_agents(brain)
    pos = _layout(brain, names)
    counts = {n: sum(1 for s in brain.agents[n].spikes if t0 <= s < t1) for n in names}
    peak = max(counts.values()) or 1

    weights = sorted(e.weight for src in brain.out_edges.values() for e in src.values() if e.weight > 0)
    cut = weights[int(edge_quantile * (len(weights) - 1))] if weights else 0.0

    fig, ax = plt.subplots(figsize=(10, 10))
    for src, targets in brain.out_edges.items():
        for dst, e in targets.items():
            if e.weight < cut or e.weight <= 0:
                continue
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            ax.plot([x0, x1], [y0, y1], color="#999999", alpha=0.08, linewidth=0.6, zorder=1)

    xs = [pos[n][0] for n in names]
    ys = [pos[n][1] for n in names]
    sizes = [30 + 600 * counts[n] / peak for n in names]
    colors = [MODULE_COLORS.get(brain.agents[n].module, "#000000") for n in names]
    ax.scatter(xs, ys, s=sizes, c=colors, edgecolors="black", linewidths=0.3, zorder=2)

    for n in names:
        if counts[n] >= 0.5 * peak and counts[n] > 0:
            x, y = pos[n]
            ax.annotate(n, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=6)

    seen = sorted({brain.agents[n].module for n in names})
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=MODULE_COLORS.get(m, "#000"), label=m)
               for m in seen]
    ax.legend(handles=handles, loc="lower left", fontsize=7, frameon=False)
    ax.set_title(title or f"Activity {t0}-{t1} ms  (node size = spikes)")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def activity_gif(brain: Brain, path: Path, *, window: int = 10, step: int = 10, fps: int = 10) -> Path:
    """Animate the graph snapshot across the run, one frame per ``step`` ms."""
    plt = _plt()
    from matplotlib import animation

    names = _ordered_agents(brain)
    pos = _layout(brain, names)
    spikes = {n: brain.agents[n].spikes for n in names}
    colors = [MODULE_COLORS.get(brain.agents[n].module, "#000000") for n in names]
    xs = [pos[n][0] for n in names]
    ys = [pos[n][1] for n in names]

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter(xs, ys, s=30, c=colors, edgecolors="black", linewidths=0.3)
    title = ax.set_title("")
    ax.set_aspect("equal")
    ax.axis("off")

    frames = list(range(0, max(1, brain.tick - window), step))

    def update(t0: int):
        sizes = []
        for n in names:
            c = sum(1 for s in spikes[n] if t0 <= s < t0 + window)
            sizes.append(30 + 200 * c)
        scat.set_sizes(sizes)
        title.set_text(f"{t0} ms")
        return scat, title

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def similarity_heatmap(matrix: Dict[str, Dict[str, float]], path: Path, title: str = "") -> Path:
    """Heatmap of modality-to-modality response similarity (cosine, early window)."""
    plt = _plt()
    keys = list(matrix)
    data = [[matrix[a][b] for b in keys] for a in keys]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center",
                    color="white" if data[i][j] < 0.6 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title(title or "Response similarity between senses")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

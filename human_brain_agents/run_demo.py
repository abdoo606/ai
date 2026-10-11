"""Multi-sensory experiment on the brain model.

Usage (from the repository root):
    PYTHONPATH=. python human_brain_agents/run_demo.py [--reps N] [--figures DIR] [--csv FILE]

It runs visual, auditory and tactile stimuli and reports:
  1. decisions (confusion matrix), reaction times, and motor latency;
  2. response distinctness (early-wave similarity between senses);
  3. detection thresholds per sense;
  4. a figure set: similarity heatmap, activity heatmap, graph snapshot and GIF.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from human_brain_agents import build_hcp_brain
from human_brain_agents.stimuli import (
    MODALITY_MODULE,
    Stimulus,
    confusion,
    detection_threshold,
    present,
    run_protocol,
    similarity_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5, help="repetitions per sense")
    parser.add_argument("--figures", default="human_brain_agents/figures",
                        help="folder for figures ('' to skip)")
    parser.add_argument("--csv", help="write a spike raster (agent, module, tick)")
    args = parser.parse_args()

    make = build_hcp_brain
    modalities = list(MODALITY_MODULE)

    # 1. Decisions and timing: stimuli in a fixed interleaved order
    brain = make()
    order = [m for _ in range(args.reps) for m in modalities]
    results = run_protocol(brain, [Stimulus(m) for m in order])
    print(f"Regions: {len(brain.agents) - 1}  + decision agent")
    print("\nDecisions (rows = presented, columns = decided):")
    table = confusion(results)
    header = " " * 10 + "".join(f"{c:>11}" for c in list(MODALITY_MODULE) + ["none"])
    print(header)
    for m in modalities:
        print(f"{m:<10}" + "".join(f"{table[m][c]:>11}" for c in list(MODALITY_MODULE) + ["none"]))

    print("\nTiming (ms from stimulus onset):")
    print(f"{'sense':<10}{'reaction':>16}{'first sensory':>16}{'first motor':>14}")
    for m in modalities:
        rts = [r.reaction_time for r in results if r.modality == m and r.reaction_time is not None]
        sens = [r.first_sensory_spike for r in results if r.modality == m and r.first_sensory_spike is not None]
        mot = [r.first_motor_spike for r in results if r.modality == m and r.first_motor_spike is not None]
        rt_s = f"{statistics.mean(rts):.1f} +/- {statistics.pstdev(rts):.1f}" if rts else "no decision"
        print(f"{m:<10}{rt_s:>16}"
              f"{statistics.mean(sens) if sens else float('nan'):>16.1f}"
              f"{statistics.mean(mot) if mot else float('nan'):>14.1f}")

    # 2. Distinctness of responses
    sm = similarity_matrix(make, modalities, repetitions=args.reps)
    print("\nResponse similarity, first 10 ms (1.0 = identical):")
    print(" " * 10 + "".join(f"{m:>11}" for m in modalities))
    for a in modalities:
        print(f"{a:<10}" + "".join(f"{sm[a][b]:>11.3f}" for b in modalities))

    # 3. Detection thresholds
    print("\nDetection threshold (lowest intensity that produces a decision):")
    grid = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
    for m in modalities:
        t = detection_threshold(make, m, grid)
        print(f"  {m:<10} {t if t is not None else 'none in grid'}")

    # 4. Figures
    if args.figures:
        from human_brain_agents.viz import activity_gif, activity_heatmap, graph_snapshot, similarity_heatmap

        out = Path(args.figures)
        demo = make()
        for m in modalities:
            present(demo, Stimulus(m), start=demo.tick)
            demo.run(400)
        print("\nFigures:")
        print("  " + str(similarity_heatmap(sm, out / "similarity_heatmap.png")))
        print("  " + str(activity_heatmap(demo, out / "activity_heatmap.png",
                                         title="Visual, auditory, then tactile stimulus")))
        print("  " + str(graph_snapshot(demo, 0, 30, out / "graph_0_30ms.png")))
        print("  " + str(activity_gif(demo, out / "activity.gif", window=10, step=10)))

    if args.csv:
        full = make()
        run_protocol(full, [Stimulus(m) for m in modalities])
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["agent", "module", "tick"])
            for name, agent in full.agents.items():
                for t in agent.spikes:
                    w.writerow([name, agent.module, t])
        print(f"Spike raster written to {args.csv}")


if __name__ == "__main__":
    main()

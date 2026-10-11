"""Run the brain on visual stimuli and print what happens.

Usage (from the repository root):
    PYTHONPATH=. python human_brain_agents/run_demo.py [--trials N] [--csv spikes.csv]

It shows:
  1. one visual stimulus: latency to motor output and module activity;
  2. repeated stimuli with learning on: how the motor response changes;
  3. energy use and the brain's homeostatic gain.
"""

import argparse
import csv
from pathlib import Path

from human_brain_agents import build_hcp_brain

VISUAL = ["L_lateraloccipital", "R_lateraloccipital"]
MOTOR = ["L_precentral", "R_precentral"]


def stimulate(brain, ms=10, strength=1.5):
    for _ in range(ms):
        for t in VISUAL:
            brain.inject(t, strength)
        brain.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--csv", help="write a spike raster (agent, module, tick)")
    parser.add_argument("--figures", default="human_brain_agents/figures",
                        help="folder for heatmap, graph and GIF ('' to skip)")
    args = parser.parse_args()

    # 1. One stimulus
    brain = build_hcp_brain()
    stimulate(brain)
    brain.run(190)
    motor_spikes = sorted(s for m in MOTOR for s in brain.agents[m].spikes)
    print(f"Agents: {len(brain.agents)}   Connections: {sum(len(v) for v in brain.out_edges.values())}")
    print(f"Motor output after one visual stimulus: first spike at {motor_spikes[0]} ms,"
          f" {len(motor_spikes)} spikes in 200 ms")
    print("\nSpikes per module:")
    for module, count in sorted(brain.module_activity().items(), key=lambda kv: -kv[1]):
        print(f"  {module:<14} {count}")

    # 2. Learning over repeated trials
    print("\nMotor spikes per 100 ms trial (learning on):")
    brain = build_hcp_brain()
    brain.learning = True
    counts = []
    for _ in range(args.trials):
        start = brain.tick
        stimulate(brain)
        brain.run(90)
        counts.append(sum(1 for m in MOTOR for s in brain.agents[m].spikes if start <= s < start + 100))
    print("  " + " ".join(str(c) for c in counts))
    print(f"  first trial: {counts[0]}   last trial: {counts[-1]}")

    # 3. Energy and homeostasis
    print(f"\nMean energy left: {brain.mean_energy_fraction():.3%}   global gain: {brain.global_gain:.2f}")

    if args.figures:
        from human_brain_agents.viz import activity_gif, activity_heatmap, graph_snapshot

        out = Path(args.figures)
        demo = build_hcp_brain()
        demo.learning = True
        for _ in range(3):
            stimulate(demo)
            demo.run(90)
        print("\nFigures:")
        print("  " + str(activity_heatmap(demo, out / "activity_heatmap.png")))
        print("  " + str(graph_snapshot(demo, 0, 30, out / "graph_0_30ms.png")))
        print("  " + str(activity_gif(demo, out / "activity.gif", window=10, step=10)))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["agent", "module", "tick"])
            for name, agent in brain.agents.items():
                for t in agent.spikes:
                    w.writerow([name, agent.module, t])
        print(f"Spike raster written to {args.csv}")


if __name__ == "__main__":
    main()

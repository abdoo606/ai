# Brain-inspired multi-agent model (HCP connectome)

Each brain region is an **agent** with one main job. Agents are leaky
integrators with an energy budget. They exchange delayed, weighted signals
along a real structural connectome (Human Connectome Project, 82 regions),
with directed flow from sensory to motor areas. Connections learn with use,
and programmatic regulators (cerebellum, brainstem, hypothalamus) keep the
whole network calibrated and inside safe limits.

The model uses only the Python standard library. Visualisation needs `matplotlib`.

## How it works

| Part | What it does | Where |
|---|---|---|
| Region agents | One agent per region (82 from the connectome, plus cerebellum, brainstem, hypothalamus). Each has a functional module. | `regions.py` |
| Connectome | 82x82 structural matrix. Keeps the strongest 20% of each region's connections. | `connectome.py`, `data/` |
| Direction | Processing hierarchy (`MODULE_LEVEL`): sensory -> thalamus -> association -> control -> motor. Forward links at full strength, feedback at `feedback_factor` (0.5). | `connectome.py` |
| Spiking dynamics | Leaky integrate-and-fire, 2 ms delays, adaptive threshold, pooled inhibition. | `core.py` |
| Learning | Spike-timing potentiation, bounded at `w_max_factor` (3x baseline), with synaptic scaling so one pathway gains only at the expense of the others. The learning rate is gated by attention and arousal. | `core.py` (`_learn`, `_scale_inputs`) |
| Neuromodulation | `arousal` (from the amygdala and brainstem) raises excitability. `attention` (from prefrontal control) amplifies the thalamic relay and gates learning. | `core.py`, `regions.py` |
| Cerebellum (programmatic) | Every 100 ms during a task, compares motor output rate with a 30 Hz target and adjusts motor gain within 0.5 to 2.0. | `CerebellumAgent` |
| Brainstem (programmatic) | Activity guard: when brain-wide firing rises above 2 spikes/ms, pooled inhibition goes up to 4x its base, then relaxes. Keeps a tonic arousal floor. | `BrainstemAgent` |
| Hypothalamus (homeostasis) | Lowers `global_gain` when average energy falls. | `HomeostatAgent` |

## Run

```bash
PYTHONPATH=. python human_brain_agents/run_demo.py --trials 30
PYTHONPATH=. python human_brain_agents/run_demo.py --csv spikes.csv   # spike raster
pip install pytest matplotlib && pytest human_brain_agents/tests
```

The demo writes three figures to `human_brain_agents/figures/`:
- `activity_heatmap.png`: every region (grouped by module) against time.
- `graph_0_30ms.png`: the network graph, with node size showing spikes in that window.
- `activity.gif`: the graph animated over the run, in 10 ms windows.

## What is validated (24 tests)

- Connectome data: 82 labels, symmetric matrix, zero diagonal.
- Region-to-module mapping, agent types, and the sensory-to-motor path.
- Directed build: forward links are stronger than feedback links.
- Idle quiet. A visual stimulus reaches motor output (about 6 ms) and control regions. Activity fades.
- Learning strengthens the visual-to-motor response. Weights stay within bounds, and summed input stays within `synaptic_slack`. Learning off leaves weights fixed. Neuromodulators raise the learning rate.
- Cerebellum keeps motor gain inside its bounds. Brainstem caps inhibition and keeps the arousal floor.
- Salience raises arousal, and attention scales the thalamic relay.
- Energy drops with activity. Exhausted agents stop. Homeostasis lowers gain when energy is low.
- Visualisation writes files.

## Limits (please read)

- **Learning gain is moderate.** With the stable settings, the motor response rises from about 7 to about 10–11 spikes per 100 ms trial over 30 trials (roughly +30–40%). Pushing `synaptic_slack` to 1.0 gives a larger early gain, but the network then runs close to self-sustained firing (up to about 30 motor spikes, with occasional silent trials), so the defaults stay conservative.
- **Direction is imposed, not measured.** The HCP matrix is undirected and has no distances, so the direction comes from the module hierarchy, and every delay is a fixed 2 ms.
- **Only 20% of connections are kept**, and most parameters were tuned by hand on this one stimulus. The dynamics are sensitive to `coupling`, `global_inhibition`, `adapt_inc`, and the brainstem's `max_rate`.
- **Cerebellum and brainstem are programmatic regulators**, not measured anatomy. The brainstem has no connectome data at all, and the cerebellum is wired by hand.
- **Not a biological simulation of a human brain.** A full 86-billion-neuron model is not feasible on ordinary hardware, and this prototype does not claim to outperform the human brain. It is an architecture for studying how specialised agents interact, on a real anatomical wiring diagram.

## Data

See `data/SOURCE.md`. The matrix comes from the MICA-MNI/ENIGMA Toolbox (BSD 3-Clause, `data/ENIGMA_LICENSE.txt`), derived from HCP S1200 data.

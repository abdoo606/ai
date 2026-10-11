# Brain-inspired multi-agent model (HCP connectome)

Each brain region is an **agent** with one main job. Agents are leaky
integrators with an energy budget. They exchange delayed, weighted signals
along a real structural connectome (Human Connectome Project, 82 regions).
Some connections learn with use, and a global controller keeps activity stable.

No external packages are needed. Only the Python standard library is used.

## How it works

| Part | What it does | Where |
|---|---|---|
| Region agents | One agent per cortical or subcortical region (84 total with the cerebellum and hypothalamus). Each has a module (visual, motor, memory, emotion, control, striatum, thalamus, association...). | `regions.py` |
| Connectome | 82x82 structural connectivity matrix. The strongest 20% of each region's connections are kept. Weights are scaled so the strongest connection = `coupling`. | `connectome.py`, `data/` |
| Spiking dynamics | Leaky integrate-and-fire with delays (2 ms by default). Adaptive threshold. Pooled inhibition (stands in for inhibitory interneurons). | `core.py` |
| Learning | Spike-timing rule: a pre-synaptic spike shortly before a post-synaptic spike strengthens the link. Weights stay within `w_max_factor` times their baseline and drift back when unused. | `core.py` (`_learn`) |
| Neuromodulation | `arousal` (raised by the amygdala) increases excitability. `attention` (raised by prefrontal control) amplifies the thalamic relay. | `core.py`, `regions.py` |
| Energy and homeostasis | Each agent has an energy budget that is spent on spikes and on time. The hypothalamus-like agent lowers `global_gain` when average energy is low. | `regions.py` (`HomeostatAgent`) |

## Run

```bash
PYTHONPATH=. python human_brain_agents/run_demo.py --trials 30
PYTHONPATH=. python human_brain_agents/run_demo.py --csv spikes.csv   # export spike raster
pip install pytest && pytest human_brain_agents/tests
```

## What is validated (tests)

- Connectome data: 82 labels, symmetric matrix, zero diagonal.
- Region-to-module mapping, and agent type per role.
- Quiet when idle. A visual stimulus reaches motor output (about 6 ms) and
  control regions. Activity fades after the stimulus and does not keep going.
- Learning makes the visual-to-motor response stronger, and weights stay in bounds.
  With learning off, weights do not change.
- Salience (amygdala) raises arousal. Attention scales the thalamic relay.
- Energy drops with activity. Exhausted agents stop. Homeostasis lowers gain when energy is low.

## Limits (please read)

- **Connectivity is undirected.** The HCP matrix has no direction and no
  distances, so both directions get the same weight and the delay is a fixed 2 ms.
- **Only 20% of connections are kept**, and the strengths were tuned by hand.
  The dynamics are sensitive to `coupling`, `global_inhibition`, and `adapt_inc`.
  Dense wiring with these settings ran away into self-sustained firing.
- **Cerebellum and hypothalamus are not in the connectome.** They are wired by
  hand. The brainstem is not modeled.
- **Learning is modest.** Motor response roughly grows from 8 to 14 spikes per
  100 ms trial over 30 trials, and it is capped by design.
- **Not a biological simulation of a human brain.** A full 86-billion-neuron
  model is not feasible on ordinary hardware, and this prototype does not
  claim to outperform the human brain. It is an architecture for studying how
  specialised agents interact, with a real anatomical wiring diagram.

## Data

See `data/SOURCE.md`. The matrix comes from the MICA-MNI/ENIGMA Toolbox
(BSD 3-Clause, `data/ENIGMA_LICENSE.txt`) and was derived from HCP S1200 data.

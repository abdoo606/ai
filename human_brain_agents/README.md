# Brain-inspired multi-agent model (HCP connectome)

Each brain region is an **agent** with one job. Agents are leaky integrators
with an energy budget. They exchange delayed, weighted signals along a real
structural connectome (Human Connectome Project, 75 regions after removing the
emotion circuit). Signals flow from sensory to motor areas. Connections learn
with use. Programmatic regulators keep the network calibrated. A decision agent
turns sensory evidence into one committed action.

Only the Python standard library is needed. `matplotlib` is needed for figures.

## Structure

| Part | What it does | Where |
|---|---|---|
| Region agents | 75 regions from the connectome, each with a functional module (visual, auditory, somatosensory, motor, memory, control, striatum, thalamus, association). | `regions.py` |
| Removed | Emotion circuit: amygdala, orbitofrontal cortex, insula, anterior cingulate. Dropped with their connections. | `regions.py` (`REMOVED_REGIONS`) |
| Connectome | 82x82 structural matrix; the removed regions are dropped and the strongest 20% of each region's connections are kept. | `connectome.py`, `data/` |
| Direction | Processing hierarchy: sensory -> thalamus -> association -> control -> motor. Forward links at full strength, feedback at 0.5x. | `connectome.py` |
| Spiking dynamics | Leaky integrate-and-fire, 2 ms delays, adaptive threshold, pooled inhibition. | `core.py` |
| Learning | Spike-timing potentiation, capped at 3x baseline, with synaptic scaling (summed input capped at 1.75x baseline). Learning rate gated by attention and arousal. | `core.py` |
| Neuromodulation | `arousal` raises excitability. `attention` amplifies the thalamic relay and gates learning. | `core.py` |
| Cerebellum (programmatic) | Every 100 ms during a task, sets motor gain so motor output nears a 30 Hz target (gain 0.5 to 2.0). | `regions.py` |
| Brainstem (programmatic) | Activity guard: when brain-wide firing exceeds 2 spikes/ms, pooled inhibition rises up to 4x its base. Keeps a tonic arousal floor. | `regions.py` |
| Hypothalamus (homeostasis) | Lowers global gain when average energy falls. | `regions.py` |
| Decision (basal-ganglia-like) | Accumulates evidence from each sense. Commits to one action when the leader reaches threshold 2.0 and leads the runner-up by 0.3. Then it executes by driving motor cortex, resets, and holds for 40 ms. | `decision.py` |
| Stimuli | Visual, auditory and tactile stimuli, with intensity, duration, onset and hemisphere. Each sense has a sensitivity gain (visual 1.0, auditory 2.0, tactile 1.0). | `stimuli.py` |

Action mapping: visual -> `press_1`, auditory -> `press_2`, tactile -> `press_3`.

## Run

```bash
PYTHONPATH=. python human_brain_agents/run_demo.py --reps 5
PYTHONPATH=. python human_brain_agents/run_demo.py --csv spikes.csv   # spike raster
pip install pytest matplotlib && pytest human_brain_agents/tests
```

The demo prints the decision confusion matrix, reaction and motor latencies,
response similarity, and detection thresholds. It writes figures to
`human_brain_agents/figures/`: `similarity_heatmap.png`, `activity_heatmap.png`,
`graph_0_30ms.png`, `activity.gif`.

## Results (from `run_demo.py --reps 5`)

**Decisions:** 15 of 15 correct. Each sense commits to its own action, with no confusions.

**Timing** (ms from onset; RT = reaction time to decision):

| sense | RT | first motor spike |
|---|---|---|
| visual | 9.2 ± 0.4 | 4 |
| auditory | 8.0 ± 0.0 | 6 |
| tactile | 9.0 ± 0.0 | 2 |

Tactile reaches motor cortex first, which fits the postcentral gyrus lying next to precentral cortex. Auditory is slowest to motor cortex, which fits its more distant route in this wiring.

**Response similarity** (cosine, first 10 ms):

| | visual | auditory | tactile |
|---|---|---|---|
| visual | 0.975 | 0.805 | 0.881 |
| auditory | 0.805 | 0.974 | 0.891 |
| tactile | 0.881 | 0.891 | 0.993 |

Each sense is reproducible (0.97–0.99 within itself) and distinct from the others (0.81–0.89 between). The first 10 ms carries the most identity information. Later activity spreads through the shared network and makes all senses look alike.

**Detection thresholds** (lowest intensity that produces a decision): visual 0.6, tactile 0.6, auditory 1.0 (on the raw intensity scale, with the sensitivity gain applied).

**Other checks (tests):** weak stimuli produce no decision. Stronger stimuli give faster decisions and more sensory activity. Two simultaneous senses give exactly one decision. No stimulus gives no decision. Each sense reaches its own sensory cortex first.

## Reward learning, noise, and comparison with humans

**Reward learning (decision agent).** Each channel keeps action values, updated with a
reward-prediction rule (`alpha = 0.3`, reward +1 for a correct action, -1 for a wrong one).
The decision always picks the highest-valued action, so it stays decisive.
Start: 24 of 24 correct. After the rule is reversed, the first block is 0% correct, then the
model relearns within about two trials per sense and reaches 100%
(`tests/test_brain.py::test_reward_learning_relearns_after_rule_switch`).

**Background noise (seeded).** Gaussian noise is added to every agent each tick.

| noise (SD) | accuracy over 5 seeds | RT (ms) | missed |
|---|---|---|---|
| 0.00 | 100% | 9.0 ± 0.8 | 0 |
| 0.02 | 100% | 9.0 ± 0.8 | 0 |
| 0.05 | 100% | 9.0 ± 0.8 | 0 |
| 0.10 | 96% (92–100% per seed) | 8.9 ± 0.8 | 2 of 60 |

**Comparison with human simple-reaction-time ranges (literature, approximate):**

| sense | human RT | model decision RT | model time to motor cortex |
|---|---|---|---|
| tactile | ~130–170 ms | 9.0 ms | 2 ms (fastest) |
| auditory | ~140–170 ms | 8.0 ms (fastest) | 6 ms (slowest) |
| visual | ~180–200 ms | 9.2 ms (slowest) | 4 ms |

Sources for the human ranges: [1] Thompson et al. as cited in [scirp study](https://www.scirp.org/html/4-2400003_2689.htm),
and reviews such as [HumanBenchmark](https://humanbenchmark.now/blog/auditory-vs-visual-reaction-time)
and [Biology Insights](https://biologyinsights.com/how-fast-is-a-humans-reaction-time/). The studies disagree on whether
auditory or tactile is fastest, so only "visual is slowest" is a robust human finding.

**Mismatch to fix:** the model reaches motor cortex last from the auditory side (6 ms). In humans, auditory
responses are typically faster than visual. The likely cause is the auditory route through this
connectome, which is long and weak. The auditory gain (2.0) partly hides it. Fixing this properly
means better auditory pathways, not a larger gain.

## Limits (please read)

- **Auditory sensitivity was calibrated, not measured.** The auditory pathway in this connectome is weaker, with a threshold about 2–5x the visual one. The auditory gain (2.0) is a fitted parameter that equalises the senses for the comparison. Without it, auditory trials were missed when they followed visual trials, because prior activity suppresses them. The result is "which sense responds", not "how easily each is driven".
- **Direction is imposed**, not measured. The HCP matrix is undirected and has no distances. Direction comes from the module hierarchy, and every delay is a fixed 2 ms.
- **Latency resolution is 1 ms** and there is no synaptic delay for the input itself, so first sensory spikes all read 0 ms.
- **Only 20% of connections are kept**, and most parameters were tuned by hand on these stimuli. Dynamics depend on `coupling`, `global_inhibition`, `adapt_inc`, the brainstem's `max_rate`, and the decision threshold.
- **Decision and regulators are programmatic.** The basal-ganglia-like decision agent, the cerebellum, and the brainstem follow designed rules. They are not measured anatomy, and the brainstem has no connectome data.
- **Not validated against human data.** The model has no comparison with human reaction times, psychophysics, or fMRI/EEG. The numbers above describe this model only.
- **Not a biological simulation of a human brain.** It uses point-like integrate-and-fire agents, not biophysical neurons, and it cannot run at the scale of 86 billion neurons. It is an architecture for studying how specialised agents interact, on a real anatomical wiring diagram.

## Data

See `data/SOURCE.md`. The matrix comes from the MICA-MNI/ENIGMA Toolbox (BSD 3-Clause, `data/ENIGMA_LICENSE.txt`), derived from HCP S1200 data.

# Human brain multi-agent model (prototype)

Each brain region is an **agent** with one main job. Agents are leaky
integrators with a limited energy budget. They exchange delayed, weighted
signals along a connectome-like graph, so quiet regions cost almost nothing.

## Layout
- `core.py`: `Message`, `Agent` (leaky integrator + energy), `Brain` (scheduler).
- `regions.py`: region agents (sensory, thalamus, association cortex,
  hippocampus, amygdala, prefrontal, basal ganglia, motor, cerebellum,
  hypothalamus) and the `HUMAN_PATHWAYS` wiring table.
- `run_demo.py`: sends stimuli and prints spikes, energy, and motor output.
- `tests/`: unit tests.

## Run
```bash
pip install pytest
PYTHONPATH=. python human_brain_agents/run_demo.py
pytest human_brain_agents/tests
```

## Status and limits
- The wiring is a **hand-written simplification**, not a measured human
  connectome. Replace `HUMAN_PATHWAYS` when real data is available.
- Working now: sensory -> thalamus -> association -> basal ganglia -> motor,
  plus the cerebellum error loop.
- Placeholders: amygdala, prefrontal, hippocampus, and hypothalamus have
  simple behavior and are not yet wired into the control loop.
- A full 86-billion-neuron biologically accurate simulation is not feasible
  on ordinary hardware, and this prototype does not claim to outperform the
  human brain.

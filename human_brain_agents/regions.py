"""Specialised region-agents and a simplified human brain wiring.

The wiring below is a hand-written, simplified map of major pathways
(sensory -> thalamus -> cortex -> action loop, with memory, emotion,
control and energy homeostasis). It is a starting point, not a measured
connectome. Replace the ``HUMAN_PATHWAYS`` table with real data later.
"""

from __future__ import annotations

from typing import Optional

from .core import Agent, Brain, Message


class Sensory(Agent):
    """Primary sensory cortex: encodes external stimuli."""


class Thalamus(Agent):
    """Relay and gate. Attention from prefrontal cortex raises its gain."""

    def __init__(self, name: str, **kw) -> None:
        super().__init__(name=name, **kw)
        self.gain = 1.0

    def on_input(self, msg: Message, tick: int) -> float:
        if msg.kind == "attention":
            self.gain = max(0.2, min(2.0, self.gain + msg.strength))
            return 0.0
        return super().on_input(msg, tick) * self.gain


class AssociationCortex(Agent):
    """Integrates information from many sources."""


class Hippocampus(Agent):
    """Memory: keeps a decaying trace of recent strong activity and replays it."""

    def __init__(self, name: str, **kw) -> None:
        super().__init__(name=name, **kw)
        self.trace = 0.0

    def on_input(self, msg: Message, tick: int) -> float:
        self.trace = min(1.0, self.trace + 0.2)
        return super().on_input(msg, tick)

    def on_tick(self, tick: int) -> float:
        self.trace *= 0.995
        # Replay: a weak drive proportional to the remembered trace
        return 0.02 * self.trace


class Amygdala(Agent):
    """Salience detector (placeholder): relays input to prefrontal cortex.
    Planned: boost salience for strong or threatening input."""


class BasalGanglia(Agent):
    """Action selection: strongest candidate wins, the rest are inhibited via wiring."""


class MotorCortex(Agent):
    """Produces actions. Spikes here are the brain's output."""

    def __init__(self, name: str, **kw) -> None:
        super().__init__(name=name, **kw)
        self.actions: list[int] = []

    def step(self, current: float, tick: int) -> bool:
        fired = super().step(current, tick)
        if fired:
            self.actions.append(tick)
        return fired


class Cerebellum(Agent):
    """Error correction: reduces motor gain when it sees error signals."""


class PrefrontalCortex(Agent):
    """Goals and control: sends attention signals to the thalamus."""


class Hypothalamus(Agent):
    """Energy homeostasis (placeholder). Its energy budget is tracked, but it
    is not wired to any region yet. Planned: broadcast 'inhibit' when low."""


# (src, dst, weight, delay in ms). Positive = excitatory, negative = inhibitory.
HUMAN_PATHWAYS = [
    ("sensory", "thalamus", 1.2, 2),
    ("thalamus", "association", 1.2, 3),
    ("thalamus", "amygdala", 0.4, 2),
    ("association", "hippocampus", 0.4, 3),
    ("hippocampus", "association", 0.3, 3),
    ("association", "prefrontal", 0.5, 4),
    ("prefrontal", "thalamus", 0.3, 4),          # top-down attention
    ("amygdala", "prefrontal", 0.5, 3),
    ("association", "basal_ganglia", 1.2, 3),
    ("basal_ganglia", "motor", 1.2, 3),
    ("motor", "cerebellum", 0.3, 2),
    ("cerebellum", "motor", -0.4, 2),            # error correction (inhibitory)
    ("basal_ganglia", "association", -0.2, 3),   # lateral inhibition / competition
]


def build_default_brain(seed: Optional[int] = None) -> Brain:
    """Create the default region-agent brain. ``seed`` is reserved for noise."""
    brain = Brain()
    brain.add_agent(Sensory("sensory"))
    brain.add_agent(Thalamus("thalamus"))
    brain.add_agent(AssociationCortex("association"))
    brain.add_agent(Hippocampus("hippocampus"))
    brain.add_agent(Amygdala("amygdala"))
    brain.add_agent(PrefrontalCortex("prefrontal"))
    brain.add_agent(BasalGanglia("basal_ganglia"))
    brain.add_agent(MotorCortex("motor"))
    brain.add_agent(Cerebellum("cerebellum"))
    brain.add_agent(Hypothalamus("hypothalamus", energy_budget=500.0))

    for src, dst, w, d in HUMAN_PATHWAYS:
        brain.connect(src, dst, w, d)
    return brain

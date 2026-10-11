"""Region-agent types and the functional module of each region.

Each Desikan-Killiany / subcortical region of the HCP connectome becomes one
agent. Its module (what it is responsible for) comes from its anatomical name.
Some modules have their own agent behaviour:

* RelayAgent (thalamus): gates input, gain set by prefrontal attention.
* MemoryAgent (hippocampus): keeps a decaying trace and replays it.
* SalienceAgent (amygdala): each spike raises the arousal modulator.
* ControlAgent (prefrontal): each spike raises the attention modulator.
* MotorAgent (motor cortex): its spikes are the output (actions).
* HomeostatAgent (hypothalamus, extra): sets brain-wide gain from energy.
* Everything else is a generic cortical integrator.
"""

from __future__ import annotations

import bisect
from typing import Dict, Optional, Set, Tuple

from .core import Agent, Message


class RelayAgent(Agent):
    """Thalamic relay: input is amplified when attention is high."""

    def on_input(self, msg: Message, tick: int) -> float:
        base = super().on_input(msg, tick)
        if self.brain is None:
            return base
        return base * (1.0 + self.brain.modulators["attention"])


class MemoryAgent(Agent):
    """Hippocampal memory: a decaying trace of recent activity, replayed as a weak drive."""

    def __init__(self, name: str, **kw) -> None:
        super().__init__(name, **kw)
        self.trace: float = 0.0

    def on_input(self, msg: Message, tick: int) -> float:
        self.trace = min(1.0, self.trace + 0.2)
        return super().on_input(msg, tick)

    def on_tick(self, tick: int) -> float:
        self.trace *= 0.995
        return 0.02 * self.trace


class SalienceAgent(Agent):
    """Amygdala: each spike raises arousal, which the brain broadcasts."""

    def on_fire(self, tick: int) -> None:
        if self.brain is not None:
            self.brain.raise_modulator("arousal", 0.02)


class ControlAgent(Agent):
    """Prefrontal control: each spike raises the attention modulator."""

    def on_fire(self, tick: int) -> None:
        if self.brain is not None:
            self.brain.raise_modulator("attention", 0.05)


class MotorAgent(Agent):
    """Motor cortex: its spikes are the brain's output (actions)."""

    @property
    def actions(self) -> list:
        return self.spikes


class HomeostatAgent(Agent):
    """Hypothalamus (simplified): sets the brain-wide gain from its energy level.

    When the average energy falls below ``low``, all agents receive less input,
    so the brain spends less. The gain returns to 1 when energy recovers.
    """

    def __init__(self, name: str, *, low: float = 0.3, high: float = 0.6, **kw) -> None:
        super().__init__(name, **kw)
        self.low = low
        self.high = high

    def on_tick(self, tick: int) -> float:
        if self.brain is not None:
            frac = self.brain.mean_energy_fraction()
            if frac >= self.high:
                g = 1.0
            elif frac <= self.low:
                g = 0.2
            else:
                g = 0.2 + 0.8 * (frac - self.low) / (self.high - self.low)
            self.brain.global_gain = g
        return 0.0


# --- anatomical name -> functional module ---------------------------------
# HCP labels use Desikan-Killiany names for cortex and short codes for subcortex.
SUBCORTICAL_NAMES: Dict[str, str] = {
    "accumb": "accumbens",
    "amyg": "amygdala",
    "caud": "caudate",
    "hippo": "hippocampus",
    "pal": "pallidum",
    "put": "putamen",
    "thal": "thalamus",
}

MODULES: Dict[str, Set[str]] = {
    "visual": {"lateraloccipital", "lingual", "cuneus", "pericalcarine"},
    "auditory": {"superiortemporal", "transversetemporal"},
    "somatosensory": {"postcentral", "supramarginal"},
    "motor": {"precentral", "paracentral"},
    "memory": {"hippocampus", "entorhinal", "parahippocampal"},
    "emotion": {"amygdala", "medialorbitofrontal", "lateralorbitofrontal",
                "rostralanteriorcingulate", "insula"},
    "control": {"rostralmiddlefrontal", "caudalmiddlefrontal", "frontalpole",
                "parsorbitalis", "parstriangularis", "parsopercularis",
                "superiorfrontal"},
    "striatum": {"caudate", "putamen", "pallidum", "accumbens"},
    "thalamus": {"thalamus"},
}
DEFAULT_MODULE = "association"


def parse_label(label: str) -> Tuple[str, str]:
    """Return (hemisphere, region_name) for 'L_precentral' or 'Lcaud' style labels."""
    if label[:2] in ("L_", "R_"):
        return label[0], label[2:]
    if label[:1] in ("L", "R") and label[1:] in SUBCORTICAL_NAMES:
        return label[0], SUBCORTICAL_NAMES[label[1:]]
    raise ValueError(f"unrecognised region label: {label!r}")


def module_of(region_name: str) -> str:
    for module, names in MODULES.items():
        if region_name in names:
            return module
    return DEFAULT_MODULE


def make_agent(label: str, **kw) -> Agent:
    """Create the right agent type for an HCP region label."""
    _, region = parse_label(label)
    module = module_of(region)
    if module == "thalamus":
        cls = RelayAgent
    elif region == "hippocampus":
        cls = MemoryAgent
    elif region == "amygdala":
        cls = SalienceAgent
    elif module == "control":
        cls = ControlAgent
    elif module == "motor":
        cls = MotorAgent
    else:
        cls = Agent
    return cls(label, role=region, module=module, **kw)


# --- hierarchy: which way signals flow between modules ----------------------
# Lower level = earlier in processing (sensory). Higher = later (motor).
# Connections go forward (low -> high) at full strength and feedback
# (high -> low) at a reduced strength. Same-level connections are symmetric.
MODULE_LEVEL: Dict[str, float] = {
    "visual": 0.0,
    "auditory": 0.0,
    "somatosensory": 0.0,
    "thalamus": 1.0,
    "association": 2.0,
    "memory": 2.0,
    "emotion": 3.0,
    "striatum": 3.0,
    "control": 4.0,
    "motor": 5.0,
    "cerebellum": 5.0,
}


def level_of(module: str) -> float:
    return MODULE_LEVEL.get(module, MODULE_LEVEL["association"])


class CerebellumAgent(Agent):
    """Programmatic cerebellum: calibrates motor gain against a target rate.

    Every ``window`` ms, while the sensory system is active, it compares the
    motor output rate with ``target_rate`` and nudges each motor agent's gain.
    Gain stays within ``gain_bounds``. The spiking connections still run too.
    """

    def __init__(self, name: str, *, target_rate: float = 0.03, window: int = 100,
                 learning_rate: float = 0.05, gain_bounds=(0.5, 2.0), **kw) -> None:
        super().__init__(name, **kw)
        self.target_rate = target_rate        # spikes per ms per motor agent
        self.window = window
        self.learning_rate = learning_rate
        self.gain_bounds = gain_bounds
        self.adjustments: list = []           # (tick, error) log

    def on_tick(self, tick: int) -> float:
        if self.brain is None or tick == 0 or tick % self.window != 0:
            return 0.0
        brain = self.brain
        start = tick - self.window
        motors = [a for a in brain.agents.values() if a.module == "motor"]
        sensors = [a for a in brain.agents.values()
                   if a.module in ("visual", "auditory", "somatosensory")]
        sensory = sum(_count(a.spikes, start, tick) for a in sensors)
        if sensory == 0 or not motors:
            return 0.0  # no task running, nothing to calibrate
        rate = sum(_count(m.spikes, start, tick) for m in motors) / (self.window * len(motors))
        error = self.target_rate - rate
        self.adjustments.append((tick, error))
        lo, hi = self.gain_bounds
        for m in motors:
            m.gain = min(hi, max(lo, m.gain + self.learning_rate * error / self.target_rate))
        return 0.0


class BrainstemAgent(Agent):
    """Programmatic brainstem: keeps the whole brain inside a safe activity range.

    * Activity guard: if brain-wide firing rate goes above ``max_rate``, pooled
      inhibition is raised (up to ``max_inhibition_factor`` times its base).
      It relaxes back when activity falls.
    * Arousal floor: keeps a small tonic arousal, like the brainstem's
      wake-promoting nuclei, so the brain is never fully silent when awake.
    """

    def __init__(self, name: str, *, max_rate: float = 2.0, window: int = 20,
                 max_inhibition_factor: float = 4.0, tonic_arousal: float = 0.05, **kw) -> None:
        super().__init__(name, **kw)
        self.max_rate = max_rate                       # spikes per ms, whole brain
        self.window = window
        self.max_inhibition_factor = max_inhibition_factor
        self.tonic_arousal = tonic_arousal
        self._base_inhibition: Optional[float] = None

    def on_tick(self, tick: int) -> float:
        brain = self.brain
        if brain is None:
            return 0.0
        if self._base_inhibition is None:
            self._base_inhibition = brain.global_inhibition
        start = max(0, tick - self.window)
        spikes = sum(brain.spike_counts[start:tick])
        rate = spikes / max(1, tick - start)
        base = self._base_inhibition
        cap = base * self.max_inhibition_factor
        if rate > self.max_rate:
            brain.global_inhibition = min(cap, brain.global_inhibition + base * 0.1)
        else:
            brain.global_inhibition = max(base, brain.global_inhibition - base * 0.05)
        if brain.modulators["arousal"] < self.tonic_arousal:
            brain.raise_modulator("arousal", self.tonic_arousal - brain.modulators["arousal"])
        return 0.0


def _count(spikes: list, start: int, end: int) -> int:
    """Number of spike times in [start, end). Spike lists are sorted."""
    return bisect.bisect_left(spikes, end) - bisect.bisect_left(spikes, start)

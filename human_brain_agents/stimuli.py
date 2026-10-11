"""Stimuli for each sensory modality, and a protocol to run them.

A stimulus is delivered to the primary sensory regions of its modality:

* visual   -> occipital cortex (lateral occipital, lingual, cuneus, pericalcarine)
* auditory -> superior and transverse temporal cortex
* tactile  -> somatosensory cortex (postcentral, supramarginal)

Each stimulus has an intensity, a duration, an onset time and a hemisphere
(``both``, ``L`` or ``R``). The protocol runs a sequence of stimuli, lets the
brain settle between them, and records what happened for each one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .core import Brain
from .decision import DEFAULT_ACTIONS, DecisionAgent

MODALITY_MODULE: Dict[str, str] = {
    "visual": "visual",
    "auditory": "auditory",
    "tactile": "somatosensory",
}

# Sensitivity calibration per modality. The auditory pathway in this
# connectome is weaker: its detection threshold is about 5x the visual one.
# Gains bring the modalities to comparable, supra-threshold drive, so the
# comparison is about *which* sense responds, not *how easily* it is driven.
# These values are fitted, not measured. See the README.
MODALITY_GAIN: Dict[str, float] = {
    "visual": 1.0,
    "auditory": 2.0,
    "tactile": 1.0,
}


@dataclass(frozen=True)
class Stimulus:
    modality: str
    intensity: float = 1.5     # input current per ms
    duration: int = 10         # ms
    onset: int = 0             # ms from the start of the trial
    hemisphere: str = "both"   # "both", "L" or "R"

    def __post_init__(self) -> None:
        if self.modality not in MODALITY_MODULE:
            raise ValueError(f"unknown modality {self.modality!r}")
        if self.hemisphere not in ("both", "L", "R"):
            raise ValueError("hemisphere must be 'both', 'L' or 'R'")
        if self.duration <= 0 or self.intensity < 0 or self.onset < 0:
            raise ValueError("duration must be > 0, intensity and onset >= 0")


def targets_for(brain: Brain, stimulus: Stimulus) -> List[str]:
    """Names of the regions that receive a stimulus."""
    module = MODALITY_MODULE[stimulus.modality]
    names = [n for n, a in brain.agents.items() if a.module == module]
    if stimulus.hemisphere != "both":
        names = [n for n in names if n.startswith(stimulus.hemisphere + "_")]
    if not names:
        raise ValueError(f"no regions for {stimulus}")
    return names


def present(brain: Brain, stimulus: Stimulus, start: Optional[int] = None) -> List[str]:
    """Schedule a stimulus. ``start`` is the trial start tick (default: now)."""
    base = brain.tick if start is None else start
    targets = targets_for(brain, stimulus)
    drive = stimulus.intensity * MODALITY_GAIN[stimulus.modality]
    for ms in range(stimulus.duration):
        brain.inject(targets, drive, at=base + stimulus.onset + ms)
    return targets


@dataclass
class TrialResult:
    modality: str
    onset: int                       # absolute tick of stimulus onset
    decided_action: Optional[str]    # action the brain committed to, or None
    decided_channel: Optional[str]
    reaction_time: Optional[int]     # ms from onset to decision
    first_sensory_spike: Optional[int]   # ms from onset, in the modality's regions
    first_motor_spike: Optional[int]     # ms from onset, in motor cortex
    response: Dict[str, int] = field(default_factory=dict)  # spikes per region in the response window
    correct: Optional[bool] = None   # True/False when a rule is given; None if no decision


def add_decision(brain: Brain, **kw) -> DecisionAgent:
    """Add the decision agent wired to all sensory channels and the motor cortex."""
    channels = {}
    for modality, module in MODALITY_MODULE.items():
        channels[modality] = [n for n, a in brain.agents.items() if a.module == module]
    motor = [n for n, a in brain.agents.items() if a.module == "motor"]
    agent = DecisionAgent("decision", channels=channels, motor_targets=motor, **kw)
    return brain.add_agent(agent)


def run_protocol(
    brain: Brain,
    stimuli: Sequence[Stimulus],
    *,
    gap: int = 300,
    response_window: int = 150,
    rule: Optional[Dict[str, str]] = None,
    learn: bool = False,
) -> List[TrialResult]:
    """Run stimuli one after another and record each trial.

    Each trial starts ``gap`` ms after the previous one ends. Trials are
    independent: spikes from earlier trials are not counted.

    ``rule`` maps each modality to the correct action. With ``learn=True`` the
    decision agent gets +1 for a correct action and -1 for a wrong one.
    """
    decision = brain.agents.get("decision")
    results: List[TrialResult] = []
    for stim in stimuli:
        start = brain.tick
        onset = start + stim.onset
        present(brain, stim, start=start)
        brain.run(stim.onset + stim.duration + response_window + gap)

        window = (onset, onset + response_window)
        response: Dict[str, int] = {}
        first_sensory = None
        first_motor = None
        module = MODALITY_MODULE[stim.modality]
        for name, agent in brain.agents.items():
            n = sum(1 for s in agent.spikes if window[0] <= s < window[1])
            if n:
                response[name] = n
            if agent.module == module:
                for s in agent.spikes:
                    if onset <= s < window[1]:
                        first_sensory = s - onset if first_sensory is None else min(first_sensory, s - onset)
                        break
            if agent.module == "motor":
                for s in agent.spikes:
                    if onset <= s < window[1]:
                        first_motor = s - onset if first_motor is None else min(first_motor, s - onset)
                        break

        action = channel = None
        rt = None
        if decision is not None:
            for cmd in decision.commands:
                if onset <= cmd.tick < window[1]:
                    action, channel = cmd.action, cmd.channel
                    rt = cmd.tick - onset
                    break
        correct = None
        if rule is not None and action is not None:
            correct = action == rule[stim.modality]
            if learn and decision is not None:
                decision.feedback(channel, action, 1.0 if correct else -1.0)
        elif rule is not None:
            correct = False
        results.append(TrialResult(
            modality=stim.modality, onset=onset, decided_action=action,
            decided_channel=channel, reaction_time=rt,
            first_sensory_spike=first_sensory, first_motor_spike=first_motor,
            response=response, correct=correct,
        ))
    return results


def confusion(results: Sequence[TrialResult]) -> Dict[str, Dict[str, int]]:
    """Counts of decided channel (or 'none') for each presented modality."""
    table: Dict[str, Dict[str, int]] = {m: {c: 0 for c in list(MODALITY_MODULE) + ["none"]}
                                         for m in MODALITY_MODULE}
    for r in results:
        key = r.decided_channel or "none"
        table[r.modality][key] += 1
    return table


def response_vector(result: TrialResult, names: Sequence[str]) -> List[float]:
    return [float(result.response.get(n, 0)) for n in names]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    import math
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


# --- analysis ----------------------------------------------------------------

def early_response_vector(brain: Brain, names: Sequence[str], onset: int, window: int) -> List[float]:
    """Spike counts per region in the first ``window`` ms after onset.

    The first wave of activity is the feed-forward response. It carries the
    most modality-specific information, because later waves spread through
    the shared network.
    """
    return [float(sum(1 for s in brain.agents[n].spikes if onset <= s < onset + window))
            for n in names]


def similarity_matrix(
    make_brain: Callable[[], Brain],
    modalities: Sequence[str] = tuple(MODALITY_MODULE),
    *,
    repetitions: int = 4,
    window: int = 10,
    intensity: float = 1.5,
    gap: int = 300,
) -> Dict[str, Dict[str, float]]:
    """Mean cosine similarity of early response vectors, for every pair of modalities.

    Within-modality values should be high (reproducible responses) and
    between-modality values lower (distinct responses).
    """
    brain = make_brain()
    names = sorted(brain.agents)
    vectors: Dict[str, List[List[float]]] = {m: [] for m in modalities}
    for _ in range(repetitions):
        for m in modalities:
            onset = brain.tick
            present(brain, Stimulus(m, intensity=intensity), start=onset)
            brain.run(10 + gap)
            vectors[m].append(early_response_vector(brain, names, onset, window))

    out: Dict[str, Dict[str, float]] = {}
    for a in modalities:
        out[a] = {}
        for b_ in modalities:
            vals = [cosine(x, y) for i, x in enumerate(vectors[a])
                    for j, y in enumerate(vectors[b_]) if not (a == b_ and i >= j)]
            out[a][b_] = sum(vals) / len(vals) if vals else float("nan")
    return out


def detection_threshold(
    make_brain: Callable[[], Brain],
    modality: str,
    intensities: Sequence[float],
) -> Optional[float]:
    """Lowest intensity in ``intensities`` that produces a decision for ``modality``."""
    for inten in sorted(intensities):
        brain = make_brain()
        res = run_protocol(brain, [Stimulus(modality, intensity=inten)])[0]
        if res.decided_action is not None:
            return inten
    return None

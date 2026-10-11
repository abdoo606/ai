"""Decisive action selection and command execution.

The decision agent plays the role of the basal ganglia. It takes evidence
from each sensory channel (spikes in that modality's regions), accumulates
it with a leak, and commits to one action only when:

* the leading channel reaches ``threshold``, and
* it leads the runner-up by at least ``margin`` (so it is decisive, not just first).

After committing it executes the command by driving the motor cortex, resets
all evidence, and holds for ``refractory`` ms so one stimulus gives one action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .core import Agent


@dataclass(frozen=True)
class Command:
    tick: int          # tick the command was issued
    channel: str       # sensory channel that won (e.g. "visual")
    action: str        # motor action it maps to (e.g. "press_1")
    reaction_time: int # ms from stimulus onset (set by the protocol), or -1


DEFAULT_ACTIONS: Dict[str, str] = {
    "visual": "press_1",
    "auditory": "press_2",
    "tactile": "press_3",
}


class DecisionAgent(Agent):
    """Winner-take-all action selector with evidence accumulation."""

    def __init__(
        self,
        name: str,
        *,
        channels: Dict[str, Sequence[str]],
        actions: Optional[Dict[str, str]] = None,
        motor_targets: Sequence[str] = (),
        threshold: float = 2.0,
        margin: float = 0.3,
        evidence_leak: float = 0.95,
        refractory: int = 40,
        command_strength: float = 2.0,
        alpha: float = 0.3,
        prior: float = 0.1,
        **kw,
    ) -> None:
        kw.setdefault("role", "decision")
        kw.setdefault("module", "decision")
        super().__init__(name, **kw)
        self.channels = {c: list(agents) for c, agents in channels.items()}
        self.actions = dict(actions or DEFAULT_ACTIONS)
        self.motor_targets = list(motor_targets)
        self.threshold = threshold
        self.margin = margin
        self.evidence_leak = evidence_leak
        self.refractory = refractory
        self.command_strength = command_strength
        self.evidence: Dict[str, float] = {c: 0.0 for c in self.channels}
        # Action values per channel. The default action starts with a small
        # prior, so behaviour begins as designed and then follows feedback.
        self.alpha = alpha
        self.q: Dict[str, Dict[str, float]] = {}
        all_actions = list(dict.fromkeys(self.actions.values()))
        for c in self.channels:
            default = self.actions.get(c)
            values = {default: prior} if default is not None else {}
            for a in all_actions:
                values.setdefault(a, 0.0)
            self.q[c] = values
        self.commands: List[Command] = []
        self.onsets: Dict[str, List[int]] = {}   # channel -> stimulus onsets (set by protocol)
        self._hold_until = -1

    def on_tick(self, tick: int) -> float:
        if self.brain is None:
            return 0.0
        # Spikes from the previous tick are final, so use tick - 1.
        t = tick - 1
        for ch, agents in self.channels.items():
            # Mean spikes per region in this tick, so big and small channels compare fairly.
            count = sum(1 for a in agents if self.brain.agents[a].last_spike == t)
            self.evidence[ch] = self.evidence[ch] * self.evidence_leak + count / len(agents)

        if tick < self._hold_until:
            return 0.0

        ranked = sorted(self.evidence.items(), key=lambda kv: kv[1], reverse=True)
        if len(ranked) < 2:
            return 0.0
        (best, e1), (_, e2) = ranked[0], ranked[1]
        if e1 >= self.threshold and (e1 - e2) >= self.margin:
            self._commit(best, tick)
        return 0.0

    def choose(self, channel: str) -> str:
        """Action with the highest learned value for this channel."""
        values = self.q[channel]
        return max(values, key=lambda a: values[a])

    def feedback(self, channel: str, action: str, reward: float) -> None:
        """Reinforce the action taken for a channel (reward-prediction update)."""
        q = self.q[channel]
        q[action] += self.alpha * (reward - q[action])

    def _commit(self, channel: str, tick: int) -> None:
        action = self.choose(channel)
        self.commands.append(Command(tick=tick, channel=channel, action=action, reaction_time=-1))
        if self.motor_targets:
            self.brain.inject(self.motor_targets, self.command_strength, at=tick + 1)
        for c in self.evidence:
            self.evidence[c] = 0.0
        self._hold_until = tick + self.refractory

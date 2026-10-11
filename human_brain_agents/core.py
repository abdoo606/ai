"""Core primitives: messages, agents, and the brain scheduler.

Time advances in discrete ticks (1 tick = 1 ms of biological time).
Messages travel along weighted edges and arrive after a delay. Agents only
do work when they receive input or have internal drive, which keeps the
simulation event-driven and cheap when the brain is quiet.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Message:
    """A spike-like signal from one region to another."""

    src: str
    dst: str
    strength: float
    kind: str = "signal"  # e.g. "signal", "inhibit", "reward", "error"
    payload: Optional[object] = None


@dataclass
class Edge:
    dst: str
    weight: float      # >0 excitatory, <0 inhibitory
    delay: int = 1     # ticks


@dataclass
class Agent:
    """A region-agent: leaky integrator with an energy budget.

    Subclasses override ``on_input`` / ``on_tick`` to define their function.
    """

    name: str
    threshold: float = 1.0
    leak: float = 0.9              # membrane decay per tick
    energy_budget: float = 1000.0  # total energy available
    cost_per_spike: float = 1.0    # energy cost per emitted spike
    base_cost: float = 0.01        # energy cost per tick, even when idle
    refractory: int = 2            # ticks after firing before it can fire again

    def __post_init__(self) -> None:
        self.potential: float = 0.0
        self.energy: float = self.energy_budget
        self.last_spike: int = -10**9
        self.spike_count: int = 0
        self.log: List[Tuple[int, str]] = []

    # --- hooks -----------------------------------------------------------
    def on_input(self, msg: Message, tick: int) -> float:
        """Convert an incoming message to input current. Override as needed."""
        if msg.kind == "inhibit":
            return -abs(msg.strength)
        return msg.strength

    def on_tick(self, tick: int) -> float:
        """Internal drive per tick (e.g. a pacemaker). Default: none."""
        return 0.0

    # --- engine ----------------------------------------------------------
    @property
    def is_exhausted(self) -> bool:
        return self.energy <= 0.0

    def step(self, current: float, tick: int) -> bool:
        """Integrate one tick. Returns True if the agent fired."""
        self.energy -= self.base_cost
        if self.is_exhausted:
            self.potential = 0.0
            return False

        self.potential = self.potential * self.leak + current
        if tick - self.last_spike <= self.refractory:
            self.potential = min(self.potential, 0.0)
            return False

        if self.potential >= self.threshold:
            self.potential = 0.0
            self.last_spike = tick
            self.spike_count += 1
            self.energy -= self.cost_per_spike
            return True
        return False


class Brain:
    """Holds agents and the connectome, and runs the simulation loop."""

    def __init__(self) -> None:
        self.agents: Dict[str, Agent] = {}
        self.edges: Dict[str, List[Edge]] = defaultdict(list)
        self.tick: int = 0
        # Scheduled deliveries: tick -> list of (dst, Message)
        self._queue: Dict[int, List[Tuple[str, Message]]] = defaultdict(list)
        self.history: List[Dict[str, int]] = []

    # --- construction ----------------------------------------------------
    def add_agent(self, agent: Agent) -> Agent:
        if agent.name in self.agents:
            raise ValueError(f"duplicate agent name: {agent.name}")
        self.agents[agent.name] = agent
        return agent

    def connect(self, src: str, dst: str, weight: float, delay: int = 1) -> None:
        if src not in self.agents or dst not in self.agents:
            raise KeyError(f"unknown agent in connection {src}->{dst}")
        if delay < 1:
            raise ValueError("delay must be >= 1 tick")
        self.edges[src].append(Edge(dst=dst, weight=weight, delay=delay))

    def inject(self, dst: str, strength: float, kind: str = "signal") -> None:
        """Send an external stimulus (e.g. from the environment) to an agent."""
        if dst not in self.agents:
            raise KeyError(dst)
        self._queue[self.tick].append(
            (dst, Message(src="environment", dst=dst, strength=strength, kind=kind))
        )

    # --- simulation ------------------------------------------------------
    def _deliver(self, tick: int) -> Dict[str, List[Message]]:
        inbox: Dict[str, List[Message]] = defaultdict(list)
        for dst, msg in self._queue.pop(tick, []):
            inbox[dst].append(msg)
        return inbox

    def step(self) -> Dict[str, int]:
        """Advance one tick. Returns {agent_name: 1} for every agent that fired."""
        t = self.tick
        inbox = self._deliver(t)
        fired: Dict[str, int] = {}

        for name, agent in self.agents.items():
            if agent.is_exhausted:
                continue
            current = agent.on_tick(t)
            for msg in inbox.get(name, []):
                current += agent.on_input(msg, t)
            if agent.step(current, t):
                fired[name] = 1
                agent.log.append((t, "spike"))
                for edge in self.edges.get(name, []):
                    w = edge.weight
                    self._queue[t + edge.delay].append(
                        (edge.dst, Message(src=name, dst=edge.dst,
                                           strength=w, kind="signal" if w >= 0 else "inhibit"))
                    )

        self.history.append(fired)
        self.tick += 1
        return fired

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()

    # --- reporting -------------------------------------------------------
    def energy_report(self) -> Dict[str, float]:
        return {n: a.energy for n, a in self.agents.items()}

    def activity_report(self) -> Dict[str, int]:
        return {n: a.spike_count for n, a in self.agents.items()}

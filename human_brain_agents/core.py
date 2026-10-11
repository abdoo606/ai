"""Core primitives: messages, agents, and the brain scheduler.

Time advances in discrete ticks (1 tick = 1 ms of model time).

* Each ``Agent`` is a leaky integrator with an energy budget. It is one
  region (or one group of neurons) with a main job.
* Agents exchange delayed, weighted messages along the edges of a graph.
* Excitatory edges marked ``plastic`` are strengthened with a simple
  spike-timing rule (pre-before-post potentiation, a Hebbian-style rule).
* Neuromodulators (``arousal`` and ``attention``) are broadcast values
  that agents can raise or read. ``arousal`` raises excitability everywhere.
  They decay over time.
* ``global_gain`` is set by a homeostatic agent. It scales all input when
  the brain runs low on energy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Message:
    """A spike-like signal from one agent to another."""

    src: str
    dst: str
    strength: float
    kind: str = "signal"  # "signal" (excitatory) or "inhibit"


@dataclass
class Edge:
    weight: float      # >0 excitatory, <0 inhibitory
    delay: int = 1     # ticks
    plastic: bool = False
    base: float = 0.0  # baseline weight; plasticity moves weight around it


class Agent:
    """A region-agent. Subclasses override the hooks to define their job."""

    def __init__(
        self,
        name: str,
        *,
        role: str = "generic",
        module: str = "association",
        threshold: float = 1.0,
        leak: float = 0.9,
        energy_budget: float = 1000.0,
        cost_per_spike: float = 1.0,
        base_cost: float = 0.01,
        refractory: int = 2,
        adapt_inc: float = 1.5,
        adapt_decay: float = 0.98,
    ) -> None:
        self.name = name
        self.role = role
        self.module = module
        self.threshold = threshold
        self.leak = leak
        self.energy_budget = energy_budget
        self.cost_per_spike = cost_per_spike
        self.base_cost = base_cost
        self.refractory = refractory
        self.adapt_inc = adapt_inc        # threshold rise after each spike
        self.adapt_decay = adapt_decay    # per-tick decay of that rise

        self.adaptation: float = 0.0
        self.potential: float = 0.0
        self.energy: float = energy_budget
        self.last_spike: int = -10**9
        self.spike_count: int = 0
        self.spikes: List[int] = []
        self.gain: float = 1.0          # local gain, may be changed by the agent
        self.brain: Optional["Brain"] = None

    # --- hooks (override in subclasses) ----------------------------------
    def on_input(self, msg: Message, tick: int) -> float:
        """Convert an incoming message to input current."""
        if msg.kind == "inhibit":
            return -abs(msg.strength)
        return msg.strength

    def on_tick(self, tick: int) -> float:
        """Internal drive per tick (e.g. a pacemaker). Default: none."""
        return 0.0

    def on_fire(self, tick: int) -> None:
        """Called right after the agent fires. Can change brain modulators."""

    # --- engine ----------------------------------------------------------
    @property
    def is_exhausted(self) -> bool:
        return self.energy <= 0.0

    @property
    def energy_fraction(self) -> float:
        return max(0.0, self.energy / self.energy_budget)

    def step(self, current: float, tick: int) -> bool:
        """Integrate one tick. Returns True if the agent fired."""
        self.energy -= self.base_cost
        if self.is_exhausted:
            self.potential = 0.0
            return False

        self.adaptation *= self.adapt_decay
        self.potential = self.potential * self.leak + current
        if tick - self.last_spike <= self.refractory:
            self.potential = min(self.potential, 0.0)
            return False

        if self.potential >= self.threshold + self.adaptation:
            self.potential = 0.0
            self.adaptation += self.adapt_inc
            self.last_spike = tick
            self.spike_count += 1
            self.spikes.append(tick)
            self.energy -= self.cost_per_spike
            return True
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, module={self.module!r})"


class Brain:
    """Holds the agents and the connectome, and runs the simulation loop."""

    MODULATORS = ("arousal", "attention")
    AROUSAL_GAIN = 0.5  # arousal 1.0 raises every agent's input by 50%

    def __init__(
        self,
        *,
        stdp_window: int = 10,
        ltp_rate: float = 0.05,
        ltd_rate: float = 0.02,
        w_max_factor: float = 3.0,
        synaptic_slack: float = 0.75,
        neuromodulated_learning: float = 0.5,
        modulator_decay: float = 0.995,
        global_inhibition: float = 0.5,
    ) -> None:
        self.agents: Dict[str, Agent] = {}
        self.out_edges: Dict[str, Dict[str, Edge]] = defaultdict(dict)
        self.in_sources: Dict[str, List[str]] = defaultdict(list)
        self.tick: int = 0
        self.modulators: Dict[str, float] = {m: 0.0 for m in self.MODULATORS}
        self.modulator_decay = modulator_decay
        self.global_gain: float = 1.0
        # Pooled inhibition: each spike in the previous tick subtracts this
        # amount (divided by the number of agents) from every agent's input.
        # It stands in for inhibitory interneurons and keeps activity stable.
        self.global_inhibition = global_inhibition
        self.inhibition_decay = 0.9        # time constant of the inhibitory pool
        self._inhibitory_trace = 0.0
        self.spike_counts: List[int] = []  # total spikes per tick (index = tick)
        self.stdp_window = stdp_window
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.w_max_factor = w_max_factor
        # Synaptic scaling: the summed plastic input to a region may exceed its
        # baseline sum by at most this fraction. Strengthening one input then
        # takes share from the others, which keeps learning stable.
        self.synaptic_slack = synaptic_slack
        # Attention and arousal gate plasticity, as neuromodulators do in the
        # brain: learning rate is multiplied by 1 + neuromodulated_learning * (attention + arousal).
        self.neuromodulated_learning = neuromodulated_learning
        self.learning: bool = True
        self._queue: Dict[int, List[Tuple[str, Message]]] = defaultdict(list)

    # --- construction ----------------------------------------------------
    def add_agent(self, agent: Agent) -> Agent:
        if agent.name in self.agents:
            raise ValueError(f"duplicate agent name: {agent.name}")
        agent.brain = self
        self.agents[agent.name] = agent
        return agent

    def connect(self, src: str, dst: str, weight: float, delay: int = 1,
                plastic: bool = False) -> None:
        if src not in self.agents or dst not in self.agents:
            raise KeyError(f"unknown agent in connection {src}->{dst}")
        if src == dst:
            raise ValueError("self-connections are not supported")
        if delay < 1:
            raise ValueError("delay must be >= 1 tick")
        if dst in self.out_edges[src]:
            raise ValueError(f"edge {src}->{dst} already exists")
        self.out_edges[src][dst] = Edge(weight=weight, delay=delay, plastic=plastic,
                                        base=weight)
        self.in_sources[dst].append(src)

    def edge(self, src: str, dst: str) -> Edge:
        return self.out_edges[src][dst]

    def inject(self, dst, strength: float, kind: str = "signal",
               at: Optional[int] = None) -> None:
        """Send an external stimulus to one agent, or to several at once.

        ``dst`` can be a name or a list of names. ``at`` is the tick to deliver
        on (default: now).
        """
        names = [dst] if isinstance(dst, str) else list(dst)
        when = self.tick if at is None else at
        if when < self.tick:
            raise ValueError("cannot inject into the past")
        for name in names:
            if name not in self.agents:
                raise KeyError(name)
            self._queue[when].append(
                (name, Message(src="environment", dst=name, strength=strength, kind=kind))
            )

    # --- simulation ------------------------------------------------------
    def _deliver(self, tick: int) -> Dict[str, List[Message]]:
        inbox: Dict[str, List[Message]] = defaultdict(list)
        for dst, msg in self._queue.pop(tick, []):
            inbox[dst].append(msg)
        return inbox

    def _decay_modulators(self) -> None:
        for m in self.MODULATORS:
            self.modulators[m] = max(0.0, min(1.0, self.modulators[m] * self.modulator_decay))

    def raise_modulator(self, name: str, amount: float) -> None:
        self.modulators[name] = max(0.0, min(1.0, self.modulators[name] + amount))

    def learning_rate_factor(self) -> float:
        m = self.modulators
        return 1.0 + self.neuromodulated_learning * (m["attention"] + m["arousal"])

    def _learn(self, post: Agent, tick: int) -> None:
        """Spike-timing plasticity on plastic excitatory inputs of ``post``.

        * pre fired shortly before post  -> potentiate (Hebbian), bounded by
          ``w_max_factor * base``.
        * otherwise                      -> weak depression back toward ``base``.
        """
        gain = self.learning_rate_factor()
        for pre_name in self.in_sources[post.name]:
            edge = self.out_edges[pre_name][post.name]
            if not edge.plastic or edge.base <= 0:
                continue
            pre = self.agents[pre_name]
            dt = tick - pre.last_spike
            if 0 <= dt <= self.stdp_window:
                cap = self.w_max_factor * edge.base
                edge.weight += self.ltp_rate * gain * (cap - edge.weight)
            else:
                edge.weight += self.ltd_rate * (edge.base - edge.weight)
        self._scale_inputs(post)

    def _scale_inputs(self, post: Agent) -> None:
        plastic = [self.out_edges[s][post.name] for s in self.in_sources[post.name]
                   if self.out_edges[s][post.name].plastic
                   and self.out_edges[s][post.name].base > 0]
        if not plastic:
            return
        total = sum(e.weight for e in plastic)
        limit = (1.0 + self.synaptic_slack) * sum(e.base for e in plastic)
        if total > limit:
            factor = limit / total
            for e in plastic:
                e.weight *= factor

    def _emit(self, agent: Agent, tick: int) -> None:
        for dst, edge in self.out_edges.get(agent.name, {}).items():
            kind = "signal" if edge.weight >= 0 else "inhibit"
            self._queue[tick + edge.delay].append(
                (dst, Message(src=agent.name, dst=dst, strength=edge.weight, kind=kind))
            )

    def step(self) -> List[str]:
        """Advance one tick. Returns the names of agents that fired."""
        t = self.tick
        self._decay_modulators()
        inbox = self._deliver(t)
        fired: List[str] = []
        n = max(1, len(self.agents))
        inhibition = self.global_inhibition * self._inhibitory_trace / n

        for name, agent in self.agents.items():
            if agent.is_exhausted:
                continue
            drive = agent.on_tick(t)
            for msg in inbox.get(name, []):
                drive += agent.on_input(msg, t)
            excitability = 1.0 + self.AROUSAL_GAIN * self.modulators["arousal"]
            current = drive * agent.gain * self.global_gain * excitability - inhibition
            if agent.step(current, t):
                fired.append(name)
                if self.learning:
                    self._learn(agent, t)
                self._emit(agent, t)
                agent.on_fire(t)

        self._inhibitory_trace = self._inhibitory_trace * self.inhibition_decay + len(fired)
        self.spike_counts.append(len(fired))
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

    def module_activity(self) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for a in self.agents.values():
            out[a.module] += a.spike_count
        return dict(out)

    def mean_energy_fraction(self) -> float:
        agents = list(self.agents.values())
        if not agents:
            return 0.0
        return sum(a.energy_fraction for a in agents) / len(agents)

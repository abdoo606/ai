import pytest

from human_brain_agents import Agent, Brain, build_hcp_brain, load_structural, parse_label, module_of
from human_brain_agents.core import Message
from human_brain_agents.regions import HomeostatAgent, RelayAgent, SalienceAgent, make_agent


VISUAL = ["L_lateraloccipital", "R_lateraloccipital"]


def stimulate(brain, ms=10, strength=1.5, targets=VISUAL):
    """Deliver a stimulus lasting ``ms`` milliseconds to the given targets."""
    for _ in range(ms):
        for t in targets:
            brain.inject(t, strength)
        brain.step()


def spikes_in(agent, start, end):
    return sum(1 for s in agent.spikes if start <= s < end)


# --- data and structure -----------------------------------------------------

def test_connectome_data_is_valid():
    labels, matrix = load_structural()
    assert len(labels) == 82
    assert len(matrix) == 82
    assert all(len(r) == 82 for r in matrix)
    assert len(set(labels)) == 82
    for i in range(82):
        assert matrix[i][i] == 0
        for j in range(82):
            assert abs(matrix[i][j] - matrix[j][i]) < 1e-6


def test_build_has_region_agents_and_extras():
    brain = build_hcp_brain()
    assert len(brain.agents) == 84
    assert "cerebellum" in brain.agents
    assert "hypothalamus" in brain.agents
    labels, _ = load_structural()
    # Every connectome region keeps at least one outgoing connection.
    assert all(len(brain.out_edges[l]) >= 1 for l in labels)
    assert sum(len(v) for v in brain.out_edges.values()) > 400


def test_label_parsing_and_modules():
    assert parse_label("L_precentral") == ("L", "precentral")
    assert parse_label("Lcaud") == ("L", "caudate")
    assert parse_label("Rhippo") == ("R", "hippocampus")
    assert module_of("precentral") == "motor"
    assert module_of("hippocampus") == "memory"
    assert module_of("thalamus") == "thalamus"
    assert module_of("superiorparietal") == "association"
    with pytest.raises(ValueError):
        parse_label("X_unknown")


def test_agent_types_follow_role():
    assert isinstance(make_agent("Lthal"), RelayAgent)
    assert isinstance(make_agent("Lamyg"), SalienceAgent)
    assert make_agent("L_precentral").__class__.__name__ == "MotorAgent"
    assert make_agent("L_rostralmiddlefrontal").__class__.__name__ == "ControlAgent"


def test_connect_rejects_bad_input():
    b = Brain()
    b.add_agent(Agent("x"))
    b.add_agent(Agent("y"))
    with pytest.raises(KeyError):
        b.connect("x", "missing", 1.0)
    with pytest.raises(ValueError):
        b.connect("x", "x", 1.0)
    with pytest.raises(ValueError):
        b.connect("x", "y", 1.0, delay=0)
    b.connect("x", "y", 1.0)
    with pytest.raises(ValueError):
        b.connect("x", "y", 1.0)


# --- dynamics ----------------------------------------------------------------

def test_idle_brain_is_quiet():
    brain = build_hcp_brain()
    brain.run(200)
    assert sum(brain.activity_report().values()) == 0


def test_stimulus_reaches_motor_and_control():
    brain = build_hcp_brain()
    stimulate(brain, ms=10)
    brain.run(90)
    motor = [a for a in brain.agents.values() if a.module == "motor"]
    control = [a for a in brain.agents.values() if a.module == "control"]
    assert sum(a.spike_count for a in motor) > 0
    assert sum(a.spike_count for a in control) > 0
    first_motor = min(s for a in motor for s in a.spikes)
    assert first_motor < 30  # ms latency from visual input to motor output


def test_activity_fades_after_stimulus():
    brain = build_hcp_brain()
    stimulate(brain, ms=10)
    brain.run(290)
    early = sum(spikes_in(a, 0, 100) for a in brain.agents.values())
    late = sum(spikes_in(a, 200, 300) for a in brain.agents.values())
    assert early > 0
    assert late < early * 0.1


def test_learning_strengthens_visual_to_motor_response():
    brain = build_hcp_brain()
    brain.learning = True
    motor = ["L_precentral", "R_precentral"]
    first = last = None
    for trial in range(20):
        start = brain.tick
        stimulate(brain, ms=10)
        brain.run(90)
        count = sum(spikes_in(brain.agents[m], start, start + 100) for m in motor)
        if trial == 0:
            first = count
        last = count
    assert first > 0
    assert last > first


def test_learning_respects_weight_bounds():
    brain = build_hcp_brain()
    brain.learning = True
    for _ in range(10):
        stimulate(brain, ms=10)
        brain.run(90)
    for src, edges in brain.out_edges.items():
        for dst, e in edges.items():
            if e.plastic and e.base > 0:
                assert 0 < e.weight <= brain.w_max_factor * e.base + 1e-9


def test_no_learning_keeps_weights_fixed():
    brain = build_hcp_brain()
    brain.learning = False
    before = {(s, d): e.weight for s, ed in brain.out_edges.items() for d, e in ed.items()}
    stimulate(brain, ms=10)
    brain.run(90)
    after = {(s, d): e.weight for s, ed in brain.out_edges.items() for d, e in ed.items()}
    assert before == after


def test_salience_raises_arousal_and_thalamus_follows_attention():
    brain = build_hcp_brain()
    stimulate(brain, ms=10, targets=["Lamyg", "Ramyg"], strength=3.0)
    brain.run(20)
    assert brain.agents["Lamyg"].spike_count > 0
    assert brain.modulators["arousal"] > 0

    relay = brain.agents["Lthal"]
    brain.modulators["attention"] = 0.0
    low = relay.on_input(Message("x", "Lthal", 1.0), 0)
    brain.modulators["attention"] = 1.0
    high = relay.on_input(Message("x", "Lthal", 1.0), 0)
    assert high == pytest.approx(2 * low)


# --- energy ------------------------------------------------------------------

def test_energy_drops_with_activity():
    brain = build_hcp_brain()
    before = brain.agents["L_lateraloccipital"].energy
    stimulate(brain, ms=10)
    brain.run(40)
    assert brain.agents["L_lateraloccipital"].energy < before


def test_exhausted_agent_stops_firing():
    brain = Brain()
    a = brain.add_agent(Agent("a", energy_budget=0.05, base_cost=0.1))
    brain.run(5)
    assert a.is_exhausted
    brain.inject("a", 5.0)
    brain.run(5)
    assert a.spike_count == 0


def test_homeostat_reduces_gain_when_energy_is_low():
    brain = build_hcp_brain()
    for agent in brain.agents.values():
        agent.energy = agent.energy_budget * 0.1
    brain.step()
    assert brain.global_gain < 0.5
    for agent in brain.agents.values():
        agent.energy = agent.energy_budget
    brain.step()
    assert brain.global_gain == pytest.approx(1.0)


def test_inhibition_reduces_firing():
    def run(weight):
        b = Brain()
        b.add_agent(Agent("src", threshold=0.5))
        b.add_agent(Agent("dst", threshold=0.5))
        b.connect("src", "dst", weight)
        for t in range(50):
            if t % 3 == 0:
                b.inject("src", 1.0)
            b.step()
        return b.agents["dst"].spike_count
    assert run(-1.0) < run(1.0)

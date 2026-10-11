from human_brain_agents import Agent, Brain, build_default_brain


def test_idle_brain_is_quiet():
    brain = build_default_brain()
    brain.run(100)
    assert sum(brain.activity_report().values()) == 0


def test_stimulus_propagates_to_motor():
    brain = build_default_brain()
    for t in range(100):
        if t % 20 == 0:
            brain.inject("sensory", strength=1.5)
        brain.step()
    assert brain.agents["sensory"].spike_count > 0
    assert brain.agents["thalamus"].spike_count > 0
    assert brain.agents["motor"].spike_count > 0


def test_energy_drops_with_activity():
    brain = build_default_brain()
    before = brain.agents["association"].energy
    for t in range(100):
        brain.inject("sensory", strength=1.5) if t % 10 == 0 else None
        brain.step()
    assert brain.agents["association"].energy < before


def test_exhausted_agent_stops_firing():
    brain = Brain()
    a = brain.add_agent(Agent("a", energy_budget=0.05, base_cost=0.1))
    brain.run(5)
    assert a.is_exhausted
    brain.inject("a", 5.0)
    brain.run(5)
    assert a.spike_count == 0


def test_inhibition_reduces_firing():
    def run(weight):
        b = Brain()
        b.add_agent(Agent("src", threshold=0.5))
        b.add_agent(Agent("dst", threshold=0.5))
        b.connect("src", "dst", weight)
        for t in range(50):
            b.inject("src", 1.0) if t % 3 == 0 else None
            b.step()
        return b.agents["dst"].spike_count
    assert run(-1.0) < run(1.0)


def test_connect_rejects_bad_input():
    import pytest
    b = Brain()
    b.add_agent(Agent("x"))
    with pytest.raises(KeyError):
        b.connect("x", "missing", 1.0)
    with pytest.raises(ValueError):
        b.connect("x", "x", 1.0, delay=0)

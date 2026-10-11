import pytest

from human_brain_agents import Agent, Brain, build_hcp_brain, load_structural, parse_label, module_of
from human_brain_agents.core import Message
from human_brain_agents.regions import HomeostatAgent, RelayAgent, is_removed, make_agent


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
    assert len(brain.agents) == 76  # 75 regions + decision agent
    assert "decision" in brain.agents
    assert "cerebellum" in brain.agents
    assert "brainstem" in brain.agents
    assert "hypothalamus" in brain.agents
    labels, _ = load_structural()
    kept = [l for l in labels if not is_removed(l)]
    # Every kept connectome region keeps at least one outgoing connection.
    assert all(len(brain.out_edges[l]) >= 1 for l in kept)
    assert sum(len(v) for v in brain.out_edges.values()) > 300


def test_emotion_circuit_is_removed():
    brain = build_hcp_brain()
    assert not any(a.module == "emotion" for a in brain.agents.values())
    for name in brain.agents:
        assert "amyg" not in name and "insula" not in name
    assert is_removed("Lamyg") and is_removed("R_insula") and is_removed("L_medialorbitofrontal")
    assert not is_removed("L_precentral")


def test_label_parsing_and_modules():
    assert parse_label("L_precentral") == ("L", "precentral")
    assert parse_label("Lcaud") == ("L", "caudate")
    assert parse_label("Rhippo") == ("R", "hippocampus")
    assert module_of("precentral") == "motor"
    assert module_of("hippocampus") == "memory"
    assert module_of("thalamus") == "thalamus"
    assert module_of("insula") == "association"  # removed region falls back to default
    assert module_of("superiorparietal") == "association"
    with pytest.raises(ValueError):
        parse_label("X_unknown")


def test_agent_types_follow_role():
    assert isinstance(make_agent("Lthal"), RelayAgent)
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


def test_attention_scales_thalamus_relay():
    brain = build_hcp_brain()
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


# --- directed hierarchy ------------------------------------------------------

def test_forward_connections_stronger_than_feedback():
    from human_brain_agents.regions import level_of
    brain = build_hcp_brain(directed=True, feedback_factor=0.5)
    forward, feedback = [], []
    for src, targets in brain.out_edges.items():
        for dst, e in targets.items():
            if src.endswith("cerebellum") or dst in ("cerebellum", "brainstem", "hypothalamus"):
                continue
            ls = level_of(brain.agents[src].module)
            ld = level_of(brain.agents[dst].module)
            if ls < ld:
                forward.append(e.weight)
            elif ls > ld:
                feedback.append(e.weight)
    assert forward and feedback
    assert max(feedback) <= 0.5 * max(forward) + 1e-9


def test_undirected_build_is_symmetric_in_strength():
    brain = build_hcp_brain(directed=False)
    for src, targets in brain.out_edges.items():
        for dst, e in targets.items():
            if dst in ("cerebellum", "brainstem", "hypothalamus") or src in ("cerebellum", "brainstem", "hypothalamus"):
                continue
            if src in brain.out_edges[dst]:
                assert brain.out_edges[dst][src].weight == pytest.approx(e.weight)


def test_sensory_to_motor_path_exists():
    brain = build_hcp_brain()
    seen = {"L_lateraloccipital"}
    frontier = ["L_lateraloccipital"]
    while frontier:
        nxt = []
        for n in frontier:
            for d in brain.out_edges[n]:
                if d not in seen:
                    seen.add(d)
                    nxt.append(d)
        frontier = nxt
    assert any(brain.agents[n].module == "motor" for n in seen)


# --- learning bounds ---------------------------------------------------------

def test_synaptic_scaling_keeps_input_totals_bounded():
    brain = build_hcp_brain()
    brain.learning = True
    for _ in range(15):
        stimulate(brain, ms=10)
        brain.run(90)
    for post in brain.agents:
        plastic = [brain.out_edges[s][post] for s in brain.in_sources[post]
                   if brain.out_edges[s][post].plastic and brain.out_edges[s][post].base > 0]
        if not plastic:
            continue
        total = sum(e.weight for e in plastic)
        limit = (1 + brain.synaptic_slack) * sum(e.base for e in plastic)
        assert total <= limit + 1e-9


def test_neuromodulators_gate_learning_rate():
    brain = build_hcp_brain()
    base = brain.learning_rate_factor()
    brain.modulators["attention"] = 1.0
    assert brain.learning_rate_factor() > base


# --- cerebellum and brainstem ------------------------------------------------

def test_cerebellum_calibrates_motor_gain_within_bounds():
    brain = build_hcp_brain()
    brain.learning = True
    for _ in range(20):
        stimulate(brain, ms=10)
        brain.run(90)
    lo, hi = brain.agents["cerebellum"].gain_bounds
    for name in ("L_precentral", "R_precentral"):
        assert lo <= brain.agents[name].gain <= hi
    assert brain.agents["cerebellum"].adjustments  # it actually acted


def test_brainstem_limits_inhibition_and_keeps_arousal_floor():
    brain = build_hcp_brain()
    bs = brain.agents["brainstem"]
    bs.max_rate = 0.0  # any activity counts as too much
    base = brain.global_inhibition
    stimulate(brain, ms=10)
    brain.run(60)
    assert brain.global_inhibition <= base * bs.max_inhibition_factor + 1e-9
    assert brain.global_inhibition > base
    assert brain.modulators["arousal"] >= bs.tonic_arousal - 1e-9


# --- visualisation -----------------------------------------------------------

def test_visualisation_writes_files(tmp_path):
    pytest.importorskip("matplotlib")
    from human_brain_agents.viz import activity_heatmap, graph_snapshot
    brain = build_hcp_brain()
    stimulate(brain, ms=10)
    brain.run(60)
    assert activity_heatmap(brain, tmp_path / "h.png").exists()
    assert graph_snapshot(brain, 0, 40, tmp_path / "g.png").exists()


# --- stimuli, decisions and sensory discrimination ---------------------------

from human_brain_agents.stimuli import (  # noqa: E402
    MODALITY_MODULE, Stimulus, confusion, detection_threshold, early_response_vector,
    present, run_protocol, similarity_matrix,
)


def test_each_modality_triggers_its_own_action():
    brain = build_hcp_brain()
    stims = [Stimulus(m) for m in MODALITY_MODULE for _ in range(2)]
    results = run_protocol(brain, stims)
    for r in results:
        assert r.decided_channel == r.modality
    table = confusion(results)
    for m in MODALITY_MODULE:
        assert table[m][m] == 2


def test_each_modality_reaches_its_own_sensory_cortex_first():
    brain = build_hcp_brain()
    results = run_protocol(brain, [Stimulus("tactile")])
    r = results[0]
    assert r.first_sensory_spike is not None
    assert r.first_sensory_spike <= 1


def test_weak_stimulus_is_not_decided_and_strong_is_faster():
    def mk():
        return build_hcp_brain()
    assert run_protocol(mk(), [Stimulus("visual", intensity=0.3)])[0].decided_action is None
    weak = run_protocol(mk(), [Stimulus("visual", intensity=1.0)])[0]
    strong = run_protocol(mk(), [Stimulus("visual", intensity=3.0)])[0]
    assert weak.reaction_time is not None and strong.reaction_time is not None
    assert strong.reaction_time <= weak.reaction_time


def test_sensory_activity_grows_with_intensity():
    counts = []
    for inten in (0.5, 1.5, 3.0):
        brain = build_hcp_brain()
        r = run_protocol(brain, [Stimulus("visual", intensity=inten)])[0]
        counts.append(sum(v for k, v in r.response.items() if brain.agents[k].module == "visual"))
    assert counts[0] <= counts[1] <= counts[2]
    assert counts[2] > counts[0]


def test_simultaneous_stimuli_give_one_decisive_action():
    brain = build_hcp_brain()
    start = brain.tick
    present(brain, Stimulus("visual"), start=start)
    present(brain, Stimulus("tactile"), start=start)
    brain.run(300)
    commands = brain.agents["decision"].commands
    assert len(commands) == 1


def test_no_stimulus_no_action():
    brain = build_hcp_brain()
    brain.run(400)
    assert brain.agents["decision"].commands == []


def test_responses_are_reproducible_and_distinct():
    sm = similarity_matrix(lambda: build_hcp_brain(), repetitions=3)
    for m in MODALITY_MODULE:
        assert sm[m][m] > 0.95
    for a in MODALITY_MODULE:
        for b_ in MODALITY_MODULE:
            if a != b_:
                assert sm[a][b_] < sm[a][a]


def test_stimulus_validation():
    with pytest.raises(ValueError):
        Stimulus("smell")
    with pytest.raises(ValueError):
        Stimulus("visual", hemisphere="X")
    with pytest.raises(ValueError):
        Stimulus("visual", duration=0)


def test_detection_threshold_exists_for_visual():
    t = detection_threshold(lambda: build_hcp_brain(), "visual", [0.2, 0.6, 1.5])
    assert t is not None and t <= 0.6


def test_hemisphere_restricts_targets():
    from human_brain_agents.stimuli import targets_for
    brain = build_hcp_brain()
    left = targets_for(brain, Stimulus("auditory", hemisphere="L"))
    assert left and all(n.startswith("L_") for n in left)


# --- reward learning, noise, robustness --------------------------------------

def test_reward_learning_relearns_after_rule_switch():
    mods = list(MODALITY_MODULE)
    rule_a = {"visual": "press_1", "auditory": "press_2", "tactile": "press_3"}
    rule_b = {"visual": "press_2", "auditory": "press_3", "tactile": "press_1"}
    brain = build_hcp_brain()
    seq = [Stimulus(m) for _ in range(8) for m in mods]
    a = run_protocol(brain, seq, rule=rule_a, learn=True)
    assert sum(r.correct for r in a) == len(a)
    b_res = run_protocol(brain, seq, rule=rule_b, learn=True)
    assert b_res[0].correct is False          # the old habit is wrong after the switch
    assert sum(r.correct for r in b_res[-9:]) >= 8  # and it has been relearned by the end


def test_feedback_moves_value_toward_reward():
    brain = build_hcp_brain()
    d = brain.agents["decision"]
    before = d.q["visual"]["press_2"]
    d.feedback("visual", "press_2", 1.0)
    assert d.q["visual"]["press_2"] > before
    d.feedback("visual", "press_2", -1.0)
    d.feedback("visual", "press_2", -1.0)
    assert d.q["visual"]["press_2"] < before


def test_noise_is_reproducible_with_seed():
    def spikes(seed):
        b = build_hcp_brain(noise=0.3, seed=seed)
        b.run(200)
        return {n: tuple(a.spikes) for n, a in b.agents.items()}
    s3 = spikes(3)
    assert sum(len(v) for v in s3.values()) > 0  # noise alone drives some activity
    assert s3 == spikes(3)
    assert s3 != spikes(4)


def test_decisions_survive_moderate_noise():
    brain = build_hcp_brain(noise=0.03, seed=0)
    res = run_protocol(brain, [Stimulus(m) for m in MODALITY_MODULE for _ in range(2)])
    assert all(r.decided_channel == r.modality for r in res)

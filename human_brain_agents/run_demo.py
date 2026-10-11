"""Run a short stimulus-response demo and print the activity per region."""

from human_brain_agents import build_default_brain


def main() -> None:
    brain = build_default_brain()

    # Present a stimulus every 20 ms for 200 ms.
    for t in range(200):
        if t % 20 == 0:
            brain.inject("sensory", strength=1.5)
        brain.step()

    print(f"Ticks simulated: {brain.tick}")
    print("\nSpikes per region:")
    for name, count in brain.activity_report().items():
        print(f"  {name:<16} {count}")

    print("\nEnergy left per region:")
    for name, e in brain.energy_report().items():
        print(f"  {name:<16} {e:8.2f}")

    motor = brain.agents["motor"]
    print(f"\nMotor actions at ms: {motor.actions}")


if __name__ == "__main__":
    main()

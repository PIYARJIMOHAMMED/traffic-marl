from rl.environments.traffic_env import TrafficEnvironment

env = TrafficEnvironment()

observation, info = env.reset()

print("="*70)

print("Research Reward Test")

print("="*70)

for step in range(20):

    action = env.action_space.sample()

    observation, reward, done, truncated, info = env.step(action)

    print()

    print(f"Step {step}")

    print("Observation")

    print(observation)

    print("Reward")

    print(round(reward,2))

env.close()
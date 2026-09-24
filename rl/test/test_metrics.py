from rl.environments.traffic_env import TrafficEnvironment
from rl.evaluation.metrics import EvaluationMetrics

env = TrafficEnvironment()

metrics = EvaluationMetrics()

observation, info = env.reset()

print("="*70)

print("Evaluation Metrics Test")

print("="*70)

for step in range(100):

    action = env.action_space.sample()

    observation, reward, done, truncated, info = env.step(action)

    metrics.update(observation)

summary = metrics.summary()

print()

print("="*70)

print("Simulation Summary")

print("="*70)

for key, value in summary.items():

    print(f"{key:<25}: {value}")

env.close()


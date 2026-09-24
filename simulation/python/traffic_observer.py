import os

import traci

from observer.observer import TrafficObserver
from metrics.csv_logger import CSVLogger


SUMO_BINARY = "sumo-gui"

CONFIG = os.path.join(
    "..",
    "networks",
    "single_intersection",
    "simulation.sumocfg"
)

traci.start([SUMO_BINARY, "-c", CONFIG])

observer = TrafficObserver()
logger = CSVLogger()

print("=" * 70)
print("Traffic Observation Engine")
print("=" * 70)

step = 0

while step < 100:

    traci.simulationStep()

    observation = observer.get_observation()

    print(f"Step {step}")

    print(observation)

    print("-" * 70)
    logger.log(
        step,
        observation
    )

    step += 1

logger.close()
traci.close()
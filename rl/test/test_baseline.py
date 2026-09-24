import traci

from rl.baselines.fixed_time_controller import FixedTimeController
from rl.environments.config import EnvironmentConfig
from rl.environments.observation import ObservationBuilder

traci.start([
    EnvironmentConfig.SUMO_BINARY,
    "-c",
    str(EnvironmentConfig.SUMO_CONFIG)
])

traffic_light = traci.trafficlight.getIDList()[0]

controller = FixedTimeController(
    traffic_light,
    green_duration=15
)

observer = ObservationBuilder(traffic_light)

print("=" * 70)
print("Fixed-Time Baseline Test")
print("=" * 70)

for step in range(60):

    controller.step()

    traci.simulationStep()

    observation = observer.get_observation()

    print(
        f"Step {step:02d} | "
        f"Phase={int(observation[4])} | "
        f"Vehicles={int(observation[0])} | "
        f"Waiting={observation[2]:.0f}"
    )

traci.close()
import traci


class TrafficObserver:

    def __init__(self):

        self.tls_id = traci.trafficlight.getIDList()[0]

    def get_vehicle_count(self):

        return len(traci.vehicle.getIDList())

    def get_average_speed(self):

        vehicle_ids = traci.vehicle.getIDList()

        if len(vehicle_ids) == 0:
            return 0

        speeds = [
            traci.vehicle.getSpeed(vehicle)
            for vehicle in vehicle_ids
        ]

        return sum(speeds) / len(speeds)

    def get_waiting_time(self):

        vehicle_ids = traci.vehicle.getIDList()

        waiting = [
            traci.vehicle.getWaitingTime(vehicle)
            for vehicle in vehicle_ids
        ]

        return sum(waiting)

    def get_stopped_vehicles(self):

        vehicle_ids = traci.vehicle.getIDList()

        stopped = 0

        for vehicle in vehicle_ids:

            if traci.vehicle.getSpeed(vehicle) < 0.1:

                stopped += 1

        return stopped

    def get_signal_phase(self):

        return traci.trafficlight.getPhase(self.tls_id)

    def get_observation(self):

        return {

            "vehicle_count": self.get_vehicle_count(),

            "average_speed": round(
                self.get_average_speed(),
                2
            ),

            "waiting_time": round(
                self.get_waiting_time(),
                2
            ),

            "stopped_vehicles": self.get_stopped_vehicles(),

            "traffic_light_phase": self.get_signal_phase()

        }
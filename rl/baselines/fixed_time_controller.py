"""
=========================================================
Traffic MARL Project

Module:
Fixed-Time Traffic Signal Controller

Description:
Baseline controller that switches traffic lights
after a fixed duration.

Author:
Mohammed Piyarji

Week:
3

Task:
3.10
=========================================================
"""

import traci


class FixedTimeController:

    def __init__(self, traffic_light_id, green_duration=15):

        self.traffic_light_id = traffic_light_id

        self.green_duration = green_duration

        self.timer = 0

        self.current_phase = 0

    def step(self):

        self.timer += 1

        if self.timer >= self.green_duration:

            if self.current_phase == 0:

                self.current_phase = 2

            else:

                self.current_phase = 0

            traci.trafficlight.setPhase(
                self.traffic_light_id,
                self.current_phase
            )

            self.timer = 0
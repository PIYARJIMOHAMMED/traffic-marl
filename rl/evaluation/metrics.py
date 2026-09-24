"""
=========================================================
Traffic MARL Project

Module:
Evaluation Metrics

Description:
Collects simulation metrics for evaluation
of traffic control algorithms.

Author:
Mohammed Piyarji

Week:
3

Task:
3.11
=========================================================
"""


class EvaluationMetrics:

    def __init__(self):

        self.vehicle_counts = []

        self.waiting_times = []

        self.average_speeds = []

        self.queue_lengths = []

    # -----------------------------------------

    def update(self, observation):

        vehicle_count = observation[0]

        average_speed = observation[1]

        waiting_time = observation[2]

        queue_length = (

            observation[5]

            + observation[6]

            + observation[7]

            + observation[8]

        )

        self.vehicle_counts.append(vehicle_count)

        self.average_speeds.append(average_speed)

        self.waiting_times.append(waiting_time)

        self.queue_lengths.append(queue_length)

    # -----------------------------------------

    def average_vehicle_count(self):

        return sum(self.vehicle_counts) / len(self.vehicle_counts)

    def average_speed(self):

        return sum(self.average_speeds) / len(self.average_speeds)

    def average_waiting_time(self):

        return sum(self.waiting_times) / len(self.waiting_times)

    def average_queue(self):

        return sum(self.queue_lengths) / len(self.queue_lengths)

    def maximum_queue(self):

        return max(self.queue_lengths)

    def maximum_waiting(self):

        return max(self.waiting_times)

    # -----------------------------------------

    def summary(self):

        return {

            "Average Vehicle Count":

                round(self.average_vehicle_count(),2),

            "Average Speed":

                round(self.average_speed(),2),

            "Average Waiting Time":

                round(self.average_waiting_time(),2),

            "Average Queue Length":

                round(self.average_queue(),2),

            "Maximum Queue":

                round(self.maximum_queue(),2),

            "Maximum Waiting":

                round(self.maximum_waiting(),2)

        }
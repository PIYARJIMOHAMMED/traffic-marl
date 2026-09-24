import csv
import os


class CSVLogger:

    def __init__(self):

        os.makedirs("logs", exist_ok=True)

        self.filename = "logs/traffic_metrics.csv"

        self.file = open(
            self.filename,
            mode="w",
            newline=""
        )

        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "Step",
            "Vehicle Count",
            "Average Speed",
            "Waiting Time",
            "Stopped Vehicles",
            "Traffic Light Phase"
        ])

    def log(self, step, observation):

        self.writer.writerow([
            step,
            observation["vehicle_count"],
            observation["average_speed"],
            observation["waiting_time"],
            observation["stopped_vehicles"],
            observation["traffic_light_phase"]
        ])

    def close(self):

        self.file.close()
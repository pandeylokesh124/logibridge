import argparse
import time
import json
import numpy as np
import paho.mqtt.client as mqtt

# Sensor simulator
class Simulator:

    def __init__(self):
        print("Generates realistic cold-chain truck sensor data with three streams...")

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--anomaly', choices=['none', 'temp_drift', 'vibration', 'combined'], default='none')
        return parser.parse_args()

    # Simulate data
    def simulate(self):
        args = self.parse_args()
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect("localhost", 1883, 60)

        truck_id = "TRK-PH-017"
        temp_base, temp_std = 4.0, 0.3
        vib_base, vib_std = 0.45, 0.05
        drift_rate = 0.08

        temp_topic = f"logibridge/trucks/{truck_id}/sensors/temperature"
        vib_topic = f"logibridge/trucks/{truck_id}/sensors/vibration"
        door_topic = f"logibridge/trucks/{truck_id}/sensors/door"

        tick = 0
        while True:
            # Temperature at 1 Hz
            current_temp = np.random.normal(temp_base, temp_std)
            if args.anomaly in ['temp_drift', 'combined']:
                current_temp += (drift_rate * tick)
            client.publish(temp_topic, json.dumps({"timestamp": time.time(), "value": current_temp}))

            # Vibration at 0.5 Hz (every 2 seconds)
            if tick % 2 == 0:
                c_vib_base, c_vib_std = vib_base, vib_std
                if args.anomaly in ['vibration', 'combined']:
                    c_vib_base, c_vib_std = 1.2, 0.15
                current_vib = np.random.normal(c_vib_base, c_vib_std)
                client.publish(vib_topic, json.dumps({"timestamp": time.time(), "value": current_vib}))

            # Door discrete event (Randomized simulation)
            if np.random.rand() > 0.98:
                state = "OPEN" if np.random.rand() > 0.5 else "CLOSE"
                client.publish(door_topic, json.dumps({"timestamp": time.time(), "state": state}))

            tick += 1
            time.sleep(1)


if __name__ == "__main__":
    Simulator().simulate()

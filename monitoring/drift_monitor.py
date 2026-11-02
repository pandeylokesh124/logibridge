import numpy as np

class DriftMonitor:
    def __init__(self):
        print("Drift monitoring...")

    # Calculate psi
    def calculate_psi(expected_probs, actual_probs, bins=4):
        # Bins: [0, 0.25), [0.25, 0.50), [0.50, 0.75), [0.75, 1.0]
        expected_hist, _ = np.histogram(expected_probs, bins=bins, range=(0, 1))
        actual_hist, _ = np.histogram(actual_probs, bins=bins, range=(0, 1))

        # Avoid division by zero
        expected_pct = np.clip(expected_hist / len(expected_probs), 0.001, None)
        actual_pct = np.clip(actual_hist / len(actual_probs), 0.001, None)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        if psi > 0.25:
            print(f"[LOGIBRIDGE DRIFT ALERT] PSI={psi:.3f}")
        return psi


if __name__ == "__main__":
    drift_monitor = DriftMonitor().calculate_psi()
    print("Drift monitor ... ", drift_monitor)
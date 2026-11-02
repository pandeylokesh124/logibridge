import numpy as np


class DataPreProcessing:
    def __init__(self):
        print("Preprocessing data...")

    # Extract features
    def extract_features(temp_window, vib_window, training_stats_path="training_stats.npy"):
        # 5-sample moving average filter
        temp_smoothed = np.convolve(temp_window, np.ones(5) / 5, mode='valid')
        vib_smoothed = np.convolve(vib_window, np.ones(5) / 5, mode='valid')

        # Feature extraction
        t_mean = np.mean(temp_smoothed)
        t_std = np.std(temp_smoothed)
        t_roc = (temp_smoothed[-1] - temp_smoothed[0]) / (len(temp_smoothed) / 60.0)  # °C/min

        v_rms = np.sqrt(np.mean(vib_smoothed ** 2))
        v_peak = np.max(np.abs(vib_smoothed))
        v_kurt = np.mean((vib_smoothed - np.mean(vib_smoothed)) ** 4) / (np.std(vib_smoothed) ** 4) if np.std(
            vib_smoothed) != 0 else 0

        features = np.array([t_mean, t_std, t_roc, v_rms, v_peak, v_kurt])

        # Normalization using fixed training stats
        stats = np.load(training_stats_path, allow_pickle=True).item()
        features_normalized = (features - stats['mean']) / stats['std']

        return features_normalized

if __name__ == "__main__":
    preprocessing = DataPreProcessing().extract_features()
    print(preprocessing)
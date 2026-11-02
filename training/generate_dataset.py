import numpy as np
import os


class GenerateDataset:
    def __init__(self):
        print("Generating dataset...")

    # Generating raw data stream
    def generate_raw_stream(self, duration_mins, anomaly_mode):
        duration_secs = duration_mins * 60
        temp_base, temp_std = 4.0, 0.3
        vib_base, vib_std = 0.45, 0.05
        drift_rate = 0.08

        temps = []
        vibs = []

        for tick in range(duration_secs):
            c_temp = np.random.normal(temp_base, temp_std)
            if anomaly_mode in ['temp_drift', 'combined']:
                c_temp += (drift_rate * tick)
            temps.append(c_temp)

            if tick % 2 == 0:
                c_vib_base, c_vib_std = vib_base, vib_std
                if anomaly_mode in ['vibration', 'combined']:
                    c_vib_base, c_vib_std = 1.2, 0.15
                vibs.append(np.random.normal(c_vib_base, c_vib_std))

        return np.array(temps), np.array(vibs)

    def extract_window_features(self, temp_window, vib_window):
        temp_smoothed = np.convolve(temp_window, np.ones(5) / 5, mode='valid')
        vib_smoothed = np.convolve(vib_window, np.ones(5) / 5, mode='valid')

        t_mean = np.mean(temp_smoothed)
        t_std = np.std(temp_smoothed)
        t_roc = (temp_smoothed[-1] - temp_smoothed[0]) / (len(temp_smoothed) / 60.0)

        v_rms = np.sqrt(np.mean(vib_smoothed ** 2))
        v_peak = np.max(np.abs(vib_smoothed))
        v_kurt = np.mean((vib_smoothed - np.mean(vib_smoothed)) ** 4) / (np.std(vib_smoothed) ** 4 + 1e-6)

        return np.array([t_mean, t_std, t_roc, v_rms, v_peak, v_kurt])

    def process_stream_to_windows(self, temps, vibs, label):
        features = []
        labels = []
        window_size_t = 30
        step_size_t = 10

        for start_t in range(0, len(temps) - window_size_t + 1, step_size_t):
            end_t = start_t + window_size_t
            temp_win = temps[start_t:end_t]

            start_v = start_t // 2
            end_v = end_t // 2
            vib_win = vibs[start_v:end_v]

            feat = self.extract_window_features(temp_win, vib_win)
            features.append(feat)
            labels.append(label)

        return np.array(features), np.array(labels)

    def main(self):
        t0, v0 = self.generate_raw_stream(20, 'none')
        X0, y0 = self.process_stream_to_windows(t0, v0, 0)

        t1, v1 = self.generate_raw_stream(15, 'temp_drift')
        X1, y1 = self.process_stream_to_windows(t1, v1, 1)

        t2, v2 = self.generate_raw_stream(15, 'combined')
        X2, y2 = self.process_stream_to_windows(t2, v2, 2)

        X0_clean_subset = X0[:60]
        training_stats = {
            'mean': np.mean(X0_clean_subset, axis=0),
            'std': np.std(X0_clean_subset, axis=0) + 1e-6
        }

        # Ensure directory exists before saving
        os.makedirs('../data_pipeline', exist_ok=True)
        np.save('../data_pipeline/training_stats.npy', training_stats)

        X_all = np.vstack([X0, X1, X2])
        y_all = np.concatenate([y0, y1, y2])
        X_norm = (X_all - training_stats['mean']) / training_stats['std']

        np.save('X_data.npy', X_norm)
        np.save('y_data.npy', y_all)
        print(f"Dataset generated. X shape: {X_norm.shape}, y shape: {y_all.shape}")


if __name__ == "__main__":
    GenerateDataset().main()
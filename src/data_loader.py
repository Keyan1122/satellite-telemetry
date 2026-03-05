import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch


class TelemetryDataset(Dataset):
    """
    PyTorch Dataset for sliding-window satellite telemetry data.
    Used for both training (normal-only) and testing (full sequence).
    """

    def __init__(self, data, window_size, stride = 1):
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.windows = self._create_windows()

    def _create_windows(self):
        windows = []
        for start in range(0, len(self.data) - self.window_size + 1, self.stride):
            end = start + self.window_size
            windows.append(self.data[start:end])
        return np.array(windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        return torch.tensor(window, dtype=torch.float32)

def load_telemetry_channel(data_dir, channel_id, split = "train"):
    """
    Load a single telemetry channel (.npy file).
    """
    file_path = os.path.join(data_dir, split, f"{channel_id}.npy")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Telemetry file not found: {file_path}")

    data = np.load(file_path)

    # Ensure 2D shape: (timesteps, features)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return data


def preprocess_telemetry(train_data, test_data):
    """
    Normalize telemetry using statistics from training data only.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    return train_scaled, test_scaled, scaler


def load_anomaly_labels(labels_csv_path, channel_id, sequence_length,):
    """
    Create binary anomaly label vector from labeled_anomalies.csv.
    """
    labels_df = pd.read_csv(labels_csv_path)

    row = labels_df[
        (labels_df["chan_id"] == channel_id)
    ]

    if row.empty:
        raise ValueError(f"No anomaly labels found for {channel_id}")

    anomaly_sequences = eval(row.iloc[0]["anomaly_sequences"])

    labels = np.zeros(sequence_length, dtype=int)

    for start, end in anomaly_sequences:
        labels[start:end + 1] = 1

    return labels


def build_datasets(data_dir, labels_csv_path, channel_id, window_size, stride = 1):
    """
    Full pipeline:
    - load train & test telemetry
    - normalize
    - create PyTorch datasets
    - generate anomaly labels (test only)
    """

    # Load raw telemetry
    train_data = load_telemetry_channel(data_dir, channel_id, split = "train")
    test_data = load_telemetry_channel(data_dir, channel_id, split = "test")

    # Normalize
    train_scaled, test_scaled, scaler = preprocess_telemetry(train_data, test_data)

    # Create datasets
    train_dataset = TelemetryDataset(train_scaled, window_size=window_size, stride = stride)
    test_dataset = TelemetryDataset(test_scaled, window_size=window_size, stride = stride)

    # Load anomaly labels (aligned to raw time series)
    anomaly_labels = load_anomaly_labels(labels_csv_path, channel_id, sequence_length = len(test_scaled))

    return train_dataset, test_dataset, anomaly_labels, scaler

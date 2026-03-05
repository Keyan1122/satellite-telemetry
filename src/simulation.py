import numpy as np
from typing import Literal


def inject_spike(x: np.ndarray, severity: float = 3.0) -> np.ndarray:
    """
    Injects a spike anomaly at a random time step and feature.

    Args:
        x: Input window of shape (seq_len, features)
        severity: Multiplier of feature standard deviation

    Returns:
        Modified copy of x with spike anomaly
    """

    x_aug = x.copy()
    seq_len, features = x.shape

    t = np.random.randint(0, seq_len)
    feature_idx = np.random.randint(0, features)

    std = np.std(x[:, feature_idx])
    x_aug[t, feature_idx] += severity * std

    return x_aug


def inject_drift(x: np.ndarray, severity: float = 0.1) -> np.ndarray:
    """
    Injects gradual drift across the sequence.

    Args:
        x: Input window (seq_len, features)
        severity: Drift slope magnitude
    """

    x_aug = x.copy()
    seq_len, features = x.shape

    drift = np.linspace(0, severity, seq_len)

    for f in range(features):
        x_aug[:, f] += drift

    return x_aug


def inject_noise(x: np.ndarray, severity: float = 0.1) -> np.ndarray:
    """
    Injects Gaussian noise.

    Args:
        x: Input window
        severity: Noise standard deviation
    """

    x_aug = x.copy()
    noise = np.random.normal(0, severity, size=x.shape)
    x_aug += noise

    return x_aug


def inject_anomaly(
    x: np.ndarray,
    anomaly_type: Literal["spike", "drift", "noise"],
    severity: float,
) -> np.ndarray:
    """
    Unified anomaly injection interface.
    """

    if anomaly_type == "spike":
        return inject_spike(x, severity)

    elif anomaly_type == "drift":
        return inject_drift(x, severity)

    elif anomaly_type == "noise":
        return inject_noise(x, severity)

    else:
        raise ValueError(f"Unknown anomaly_type: {anomaly_type}")
        
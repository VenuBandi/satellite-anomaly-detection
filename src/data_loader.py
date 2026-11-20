"""Download NASA SMAP/MSL telemetry data from the telemanom repository."""
import os
import requests
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
BASE_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/data"
LABEL_URL = f"{BASE_URL}/labeled_anomalies.csv"
TRAIN_URL = f"{BASE_URL}/train"
TEST_URL  = f"{BASE_URL}/test"

SMAP_CHANNELS = ["A-1", "D-14", "F-4", "G-3", "P-10"]  # representative 5


def _download_file(url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)


def download_channels(channels: list[str] = SMAP_CHANNELS):
    _download_file(LABEL_URL, DATA_DIR / "labeled_anomalies.csv")
    for ch in channels:
        _download_file(f"{TRAIN_URL}/{ch}.npy", DATA_DIR / "train" / f"{ch}.npy")
        _download_file(f"{TEST_URL}/{ch}.npy",  DATA_DIR / "test"  / f"{ch}.npy")
    print(f"Downloaded {len(channels)} channels + labels.")


def load_channel(channel: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    train = np.load(DATA_DIR / "train" / f"{channel}.npy")
    test  = np.load(DATA_DIR / "test"  / f"{channel}.npy")
    labels = pd.read_csv(DATA_DIR / "labeled_anomalies.csv")
    ch_labels = labels[labels["chan_id"] == channel].reset_index(drop=True)
    return train, test, ch_labels


def get_anomaly_mask(test_len: int, label_row: pd.Series) -> np.ndarray:
    """Convert anomaly_sequences string to boolean mask."""
    import ast
    mask = np.zeros(test_len, dtype=bool)
    if pd.isna(label_row.get("anomaly_sequences", None)):
        return mask
    for start, end in ast.literal_eval(label_row["anomaly_sequences"]):
        mask[start:end] = True
    return mask


if __name__ == "__main__":
    download_channels()
    train, test, lbls = load_channel("A-1")
    print(f"A-1  train shape: {train.shape}  test shape: {test.shape}")
    print(lbls)

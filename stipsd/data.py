"""Explicit array contracts and training-only scaling."""
import hashlib
import numpy as np


def load_data(path):
    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data["images"], dtype=np.float32)
        y = np.asarray(data["targets"], dtype=np.float32)
        ids = np.asarray(data["sample_ids"], dtype=str)
    if x.ndim != 3 or x.shape[1:] != (40, 40):
        raise ValueError("images must have shape (N,40,40), axes: sample,space,time")
    if y.shape != (len(x), 3) or ids.shape != (len(x),) or len(x) < 10:
        raise ValueError("Need at least 10 aligned samples; targets=(N,3), ids=(N,)")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or x.min() < 0:
        raise ValueError("Non-finite data or negative concentration")
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated sample IDs")
    # Reject duplicate model inputs before splitting, even if labels differ.
    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in x]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate images: group or deduplicate before splitting")
    return x, y, ids


def split_indices(n, seed=42):
    if n < 10:
        raise ValueError("At least 10 samples are required")
    order = np.random.default_rng(seed).permutation(n)
    a, b = int(n * .7), int(n * .15)
    return {"train": order[:a], "val": order[a:a+b], "test": order[a+b:]}


def fit_scaler(x_train, y_train):
    maximum = float(x_train.max())
    if maximum <= 0:
        raise ValueError("Training images contain no positive concentration")
    std = y_train.std(axis=0)
    return {
        "fitted_on": "train_only", "image_offset": 0.0,
        "image_divisor": maximum,
        "label_order": ["T", "L", "M_raw"],
        "label_mean": y_train.mean(axis=0).tolist(),
        "label_std": np.where(std > 0, std, 1.0).tolist(),
        "label_range": np.ptp(y_train, axis=0).tolist(),
        "note": "No conversion of archived mass units; no per-test-image scaling.",
    }


def transform_images(x, scaler):
    return np.clip(x / scaler["image_divisor"], 0, 1).astype(np.float32)


def normalize_targets(y, scaler):
    return ((y - scaler["label_mean"]) / scaler["label_std"]).astype(np.float32)


def inverse_targets(y, scaler):
    return y * np.asarray(scaler["label_std"]) + np.asarray(scaler["label_mean"])

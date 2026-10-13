"""Keep ordinary percentage errors separate from range-normalized errors."""
import numpy as np
from skimage.metrics import structural_similarity


def image_metrics(reference, completed, mask):
    residual = np.asarray(completed, dtype=float) - reference
    return {
        "completion_RMSE_all": float(np.sqrt(np.mean(residual**2))),
        "completion_RMSE_missing": float(np.sqrt(np.mean(residual[mask]**2))) if mask.any() else 0.0,
        "completion_SSIM_all": float(structural_similarity(reference, completed, data_range=1.0)),
    }


def source_metrics(truth, prediction, label_range):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    ranges = np.asarray(label_range)
    absolute = np.abs(prediction - truth)
    raw = np.full(3, np.nan)
    normalized = np.full(3, np.nan)
    np.divide(100 * absolute, np.abs(truth), out=raw, where=truth != 0)
    np.divide(100 * absolute, ranges, out=normalized, where=ranges > 0)
    values = {}
    for j, name in enumerate(("T", "L", "M_raw")):
        values.update({f"true_{name}": float(truth[j]), f"pred_{name}": float(prediction[j]),
                       f"AE_{name}": float(absolute[j]), f"APE_{name}_pct": float(raw[j]),
                       f"range_NAE_{name}_pct": float(normalized[j])})
    values["mean_APE_pct"] = float(raw.mean())
    values["mean_range_NAE_pct"] = float(normalized.mean())
    return values

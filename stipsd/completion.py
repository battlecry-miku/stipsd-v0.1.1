"""Geometry-only completion; fixed-rank SVD is not full CV-selected DINEOF."""
import cv2
import numpy as np

METHODS = ("no_completion", "linear", "dineof_fixed_rank", "telea", "ns_only")
SCENARIOS = ("spatial_strip", "temporal_truncation", "mixed")


def make_mask(shape, scenario, rate, rng):
    if scenario not in SCENARIOS or not 0 <= rate < 1:
        raise ValueError("Unknown scenario or rate outside [0,1)")
    mask = np.zeros(shape, dtype=bool)
    if rate == 0:
        return mask
    nx, nt = shape
    if min(shape) < 3:
        raise ValueError("Need at least three cells on each axis")
    # Axis 0 is space, axis 1 is time throughout this package.
    spatial = rate if scenario == "spatial_strip" else 1 - np.sqrt(1 - rate)
    temporal = rate if scenario == "temporal_truncation" else 1 - np.sqrt(1 - rate)
    if scenario in ("spatial_strip", "mixed"):
        count = min(nx - 1, max(1, round(nx * spatial)))
        start = int(rng.integers(nx - count + 1))
        mask[start:start+count, :] = True
    if scenario in ("temporal_truncation", "mixed"):
        count = min(nt - 1, max(1, round(nt * temporal)))
        mask[:, nt-count:] = True
    return mask


def checked_input(image, mask):
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask)
    if image.ndim != 2 or image.shape != mask.shape or mask.dtype != bool:
        raise ValueError("Expected 2D image and equally shaped boolean missing mask")
    if mask.all():
        raise ValueError("Cannot complete an image without any observed pixels")
    known = image[~mask]
    if not np.isfinite(known).all() or known.min() < 0 or known.max() > 1:
        raise ValueError("Observed pixels must be finite and in [0,1]")
    out = image.copy()
    out[mask] = 0  # Hidden reference values must never reach an algorithm.
    return out, mask


def complete(image, mask, method, rank=6, iterations=20):
    out, mask = checked_input(image, mask)
    observed = out.copy()
    if method not in METHODS:
        raise ValueError(f"Unknown completion method: {method}")
    if not mask.any() or method == "no_completion":
        return out
    if method == "linear":
        filled = ~mask.copy()
        # Interpolate along space, then time; np.interp uses nearest endpoint
        # values outside the observed interval, not forecast extrapolation.
        for column in range(out.shape[1]):
            valid = filled[:, column]
            if valid.any():
                out[:, column] = np.interp(np.arange(len(valid)), np.flatnonzero(valid), out[valid, column])
                filled[:, column] = True
        for row in range(out.shape[0]):
            valid = filled[row]
            if valid.any():
                out[row] = np.interp(np.arange(len(valid)), np.flatnonzero(valid), out[row, valid])
    elif method == "dineof_fixed_rank":
        if not 1 <= rank <= min(out.shape) or iterations < 1:
            raise ValueError("Invalid SVD rank or iteration count")
        out[mask] = out[~mask].mean()
        for _ in range(iterations):
            u, s, vt = np.linalg.svd(out, full_matrices=False)
            approximation = (u[:, :rank] * s[:rank]) @ vt[:rank]
            out[mask] = approximation[mask]
            out[~mask] = observed[~mask]
    else:
        flag = cv2.INPAINT_NS if method == "ns_only" else cv2.INPAINT_TELEA
        # OpenCV's Telea gradient update is intensity-scale sensitive on
        # [0,1] float inputs. Use its conventional 8-bit image interface;
        # restore original observed floats after the internal round trip.
        encoded = np.round(out * 255).astype(np.uint8)
        out = cv2.inpaint(encoded, mask.astype(np.uint8) * 255, 3, flag).astype(np.float32) / 255
    out = np.clip(out, 0, 1).astype(np.float32)
    out[~mask] = observed[~mask]
    return out

"""Optional source-free ADR proxy correction.

Coordinates are [sample,space,time]; u, d, k are pixel-grid coefficients
U*dt/dx, D*dt/dx**2, K*dt. Only zero-offset concentration scaling is valid.
"""
import numpy as np
import torch
from .completion import checked_input, complete


def adr_residual(field, u, d, k):
    center = field[:, 1:-1, 1:-1]
    dt = (field[:, 1:-1, 2:] - field[:, 1:-1, :-2]) / 2
    dx = (field[:, 2:, 1:-1] - field[:, :-2, 1:-1]) / 2
    dxx = field[:, 2:, 1:-1] - 2 * center + field[:, :-2, 1:-1]
    return dt + u * dx - d * dxx + k * center


def correct(image, mask, settings):
    observed, mask = checked_input(image, mask)
    required = ("u_pixel", "d_pixel", "k_pixel", "iterations", "learning_rate")
    if any(key not in settings for key in required):
        raise ValueError("ADR settings must explicitly specify coefficients and optimizer")
    if not all(np.isfinite(settings[key]) for key in required):
        raise ValueError("Non-finite ADR setting")
    if settings["d_pixel"] < 0 or settings["k_pixel"] < 0 or settings["learning_rate"] <= 0 or settings["iterations"] < 1:
        raise ValueError("Invalid ADR setting")
    initial = complete(observed, mask, "ns_only")
    if not mask.any():
        return initial
    target = torch.from_numpy(observed[None])
    missing = torch.from_numpy(mask[None])
    field = torch.from_numpy(initial[None]).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([field], lr=settings["learning_rate"])
    interior = missing[:, 1:-1, 1:-1].float()
    # Penalize jumps across observed/missing interfaces. A finite output does
    # not by itself prove convergence or mass conservation.
    edges_x = (missing[:, 1:] != missing[:, :-1]).float()
    edges_t = (missing[:, :, 1:] != missing[:, :, :-1]).float()
    for _ in range(int(settings["iterations"])):
        optimizer.zero_grad()
        projected = torch.where(missing, field, target)
        r = adr_residual(projected, settings["u_pixel"], settings["d_pixel"], settings["k_pixel"])
        loss = (r.square() * interior).sum() / interior.sum().clamp_min(1)
        boundary = (((projected[:, 1:] - projected[:, :-1]).square() * edges_x).sum()
                    + ((projected[:, :, 1:] - projected[:, :, :-1]).square() * edges_t).sum())
        loss = loss + .05 * boundary / (edges_x.sum() + edges_t.sum()).clamp_min(1)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            field.clamp_(0, 1)
            field.copy_(torch.where(missing, field, target))
    result = field.detach().numpy()[0]
    if not np.isfinite(result).all():
        raise FloatingPointError("ADR proxy produced a non-finite field")
    return result

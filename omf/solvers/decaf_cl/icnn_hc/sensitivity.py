"""Data-driven interpolation of frozen Iowa240 regulator responses."""

from __future__ import annotations

import numpy as np

from .data import RuntimeDataError


def response_for_taps(data: dict, taps: np.ndarray) -> np.ndarray:
    """Return the summed margin response for a vector of regulator tap states."""
    locations = data["regulator"]["locations"].astype(str)
    anchors = data["regulator"]["tap_anchors"].astype(float)
    library = data["regulator"]["response_margin_pu"].astype(float, copy=False)
    taps = np.asarray(taps, dtype=float)
    if taps.shape != (len(locations),):
        raise RuntimeDataError(f"Expected {len(locations)} regulator taps, got shape {taps.shape}")
    if library.shape[:2] != (len(locations), len(anchors)):
        raise RuntimeDataError("Regulator response location/anchor alignment error")
    if np.any(taps < anchors.min()) or np.any(taps > anchors.max()):
        raise ValueError(f"Regulator taps must lie in [{anchors.min():g}, {anchors.max():g}]")
    total = np.zeros(library.shape[2:], dtype=float)
    for location_index, tap in enumerate(taps):
        if tap in anchors:
            total += library[location_index, int(np.flatnonzero(anchors == tap)[0])]
            continue
        upper = int(np.searchsorted(anchors, tap))
        lower = upper - 1
        fraction = (tap - anchors[lower]) / (anchors[upper] - anchors[lower])
        total += (1.0 - fraction) * library[location_index, lower] + fraction * library[location_index, upper]
    return total


def weighted_constraint_relief(lambda_rows: np.ndarray, delta_margin: np.ndarray) -> float:
    """Return ``-sum(lambda * delta_g)`` as a candidate-screening score.

    The score ranks frozen response actions under current maximum-HC KKT
    weights. It is not reported as, or substituted for, realized HC gain.
    """
    return float(-np.sum(np.asarray(lambda_rows) * np.asarray(delta_margin)))

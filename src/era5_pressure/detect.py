from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter


@dataclass(frozen=True)
class DetectionConfig:
    spatial_window: int = 9
    background_window: int = 21
    min_anomaly_hpa: float = 2.0
    min_separation_km: float = 350.0
    max_centres_per_type: int = 12


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = np.radians([lat1, lat2])
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(6371.0 * 2 * np.arcsin(np.sqrt(a)))


def _keep_separated(candidates: list[dict], distance_km: float, limit: int) -> list[dict]:
    kept: list[dict] = []
    for item in candidates:
        if all(
            _haversine_km(item["lat"], item["lon"], old["lat"], old["lon"]) >= distance_km
            for old in kept
        ):
            kept.append(item)
            if len(kept) == limit:
                break
    return kept


def detect_pressure_centres(
    mslp: xr.DataArray,
    config: DetectionConfig = DetectionConfig(),
) -> pd.DataFrame:
    """Detect high/low centres from MSLP(time, lat, lon).

    Input may be in Pa or hPa. Output pressure and anomaly are always hPa.
    Longitudes are preserved in the source convention.
    """
    required = {"time", "lat", "lon"}
    if not required.issubset(mslp.dims):
        raise ValueError(f"MSLP dimensions must include {sorted(required)}; got {mslp.dims}")
    if config.spatial_window % 2 == 0 or config.background_window % 2 == 0:
        raise ValueError("Filter windows must be odd integers")

    rows: list[dict] = []
    lats = np.asarray(mslp.lat.values)
    lons = np.asarray(mslp.lon.values)
    for t_index, time in enumerate(mslp.time.values):
        field = np.asarray(mslp.isel(time=t_index).values, dtype=np.float64)
        if np.nanmedian(field) > 2_000:
            field = field / 100.0
        finite = np.isfinite(field)
        if not finite.any():
            continue
        fill = float(np.nanmedian(field))
        clean = np.where(finite, field, fill)
        background = uniform_filter(clean, size=config.background_window, mode=("nearest", "wrap"))
        anomaly = clean - background
        high_mask = finite & (clean == maximum_filter(clean, size=config.spatial_window, mode=("nearest", "wrap")))
        low_mask = finite & (clean == minimum_filter(clean, size=config.spatial_window, mode=("nearest", "wrap")))

        for kind, mask, sign in (("H", high_mask, 1.0), ("L", low_mask, -1.0)):
            indices = np.argwhere(mask & (sign * anomaly >= config.min_anomaly_hpa))
            candidates = [
                {
                    "time": pd.Timestamp(time),
                    "type": kind,
                    "lat": float(lats[i]),
                    "lon": float(lons[j]),
                    "pressure_hpa": float(clean[i, j]),
                    "anomaly_hpa": float(anomaly[i, j]),
                }
                for i, j in indices
            ]
            candidates.sort(key=lambda r: sign * r["anomaly_hpa"], reverse=True)
            rows.extend(_keep_separated(candidates, config.min_separation_km, config.max_centres_per_type))

    columns = ["time", "type", "lat", "lon", "pressure_hpa", "anomaly_hpa"]
    return pd.DataFrame(rows, columns=columns).sort_values(["time", "type", "lat"], ignore_index=True)


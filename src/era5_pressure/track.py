from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TrackingConfig:
    max_speed_kmh: float = 120.0
    max_gap_hours: float = 12.0
    pressure_cost_km_per_hpa: float = 15.0
    min_track_points: int = 2


def _distance_matrix(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    lat1 = np.radians(a.lat.to_numpy())[:, None]
    lat2 = np.radians(b.lat.to_numpy())[None, :]
    dlat = lat2 - lat1
    dlon = np.radians(b.lon.to_numpy()[None, :] - a.lon.to_numpy()[:, None])
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def build_trajectories(
    centres: pd.DataFrame,
    config: TrackingConfig = TrackingConfig(),
) -> pd.DataFrame:
    """Link centres with a gated Hungarian assignment, independently for H and L."""
    required = {"time", "type", "lat", "lon", "pressure_hpa", "anomaly_hpa"}
    missing = required.difference(centres.columns)
    if missing:
        raise ValueError(f"Missing centre columns: {sorted(missing)}")
    data = centres.copy()
    data["time"] = pd.to_datetime(data.time)
    data = data.sort_values(["time", "type"]).reset_index(drop=True)
    data["track_id"] = -1
    next_id = 0

    for kind in ("H", "L"):
        active: dict[int, int] = {}
        kind_indices = data.index[data.type == kind]
        for time in data.loc[kind_indices, "time"].drop_duplicates().sort_values():
            current = data.loc[(data.type == kind) & (data.time == time)]
            active = {
                tid: idx for tid, idx in active.items()
                if (time - data.at[idx, "time"]).total_seconds() / 3600 <= config.max_gap_hours
            }
            assigned_current: set[int] = set()
            if active and not current.empty:
                previous_indices = list(active.values())
                previous = data.loc[previous_indices]
                distance = _distance_matrix(previous, current)
                pressure = np.abs(
                    previous.pressure_hpa.to_numpy()[:, None] - current.pressure_hpa.to_numpy()[None, :]
                )
                cost = distance + config.pressure_cost_km_per_hpa * pressure
                hours = np.array([
                    (time - data.at[i, "time"]).total_seconds() / 3600 for i in previous_indices
                ])[:, None]
                valid = distance <= config.max_speed_kmh * hours
                gated = np.where(valid, cost, 1e12)
                for r, c in zip(*linear_sum_assignment(gated)):
                    if not valid[r, c]:
                        continue
                    idx = int(current.index[c])
                    tid = int(data.at[previous_indices[r], "track_id"])
                    data.at[idx, "track_id"] = tid
                    active[tid] = idx
                    assigned_current.add(idx)
            for idx in current.index:
                if idx not in assigned_current:
                    data.at[idx, "track_id"] = next_id
                    active[next_id] = int(idx)
                    next_id += 1

    counts = data.groupby("track_id").size()
    data = data[data.track_id.isin(counts[counts >= config.min_track_points].index)].copy()
    data["step"] = data.groupby("track_id").cumcount()
    return data.sort_values(["track_id", "time"]).reset_index(drop=True)


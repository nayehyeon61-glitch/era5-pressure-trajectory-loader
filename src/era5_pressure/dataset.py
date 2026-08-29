from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PressureTrajectoryDataset(Dataset):
    """Fixed-length H/L trajectory windows for sequence prediction.

    Each feature vector is [lat, lon, pressure_hPa, anomaly_hPa, delta_hours].
    The first ``history`` points are input and the next ``horizon`` points target.
    """

    def __init__(self, trajectories: pd.DataFrame, history: int, horizon: int):
        if history < 1 or horizon < 1:
            raise ValueError("history and horizon must be positive")
        self.history = history
        self.horizon = horizon
        self.windows: list[tuple[int, int]] = []
        self.groups: dict[int, pd.DataFrame] = {}
        length = history + horizon
        for track_id, group in trajectories.groupby("track_id"):
            group = group.sort_values("time").reset_index(drop=True)
            self.groups[int(track_id)] = group
            self.windows.extend((int(track_id), start) for start in range(len(group) - length + 1))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        track_id, start = self.windows[index]
        frame = self.groups[track_id].iloc[start : start + self.history + self.horizon]
        time = pd.to_datetime(frame.time)
        dt = (time - time.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float32) / 3600.0
        values = np.column_stack([
            frame.lat.to_numpy(np.float32),
            frame.lon.to_numpy(np.float32),
            frame.pressure_hpa.to_numpy(np.float32),
            frame.anomaly_hpa.to_numpy(np.float32),
            dt,
        ])
        sequence = torch.from_numpy(values)
        return {
            "history": sequence[: self.history],
            "target": sequence[self.history :],
            "type": torch.tensor(1 if frame.type.iloc[0] == "H" else 0),
            "track_id": torch.tensor(track_id),
        }


import numpy as np
import pandas as pd
import xarray as xr

from era5_pressure.dataset import PressureTrajectoryDataset
from era5_pressure.detect import DetectionConfig, detect_pressure_centres
from era5_pressure.track import TrackingConfig, build_trajectories


def test_moving_low_is_detected_and_tracked():
    lat = np.linspace(20, 50, 31)
    lon = np.linspace(110, 150, 41)
    time = pd.date_range("2025-01-01", periods=5, freq="6h")
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    fields = []
    for step in range(len(time)):
        centre_lon = 120 + 2 * step
        low = -18 * np.exp(-((yy - 35) ** 2 + (xx - centre_lon) ** 2) / 18)
        fields.append((1015 + low) * 100)
    mslp = xr.DataArray(
        np.stack(fields), dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}
    )
    centres = detect_pressure_centres(
        mslp,
        DetectionConfig(spatial_window=7, background_window=15, min_anomaly_hpa=2, min_separation_km=300),
    )
    lows = centres[centres.type == "L"]
    assert len(lows) == 5
    tracks = build_trajectories(lows, TrackingConfig(max_speed_kmh=150, min_track_points=2))
    assert tracks.track_id.nunique() == 1
    dataset = PressureTrajectoryDataset(tracks, history=3, horizon=2)
    sample = dataset[0]
    assert tuple(sample["history"].shape) == (3, 5)
    assert tuple(sample["target"].shape) == (2, 5)


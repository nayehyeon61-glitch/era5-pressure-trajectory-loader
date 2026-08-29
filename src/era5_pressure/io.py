from __future__ import annotations

from pathlib import Path

import xarray as xr


ALIASES = {
    "valid_time": "time",
    "latitude": "lat",
    "longitude": "lon",
    "msl": "mslp",
    "mean_sea_level_pressure": "mslp",
}


def open_era5_mslp(paths: list[str | Path], chunks: dict | None = None) -> xr.DataArray:
    ds = xr.open_mfdataset([str(p) for p in paths], combine="by_coords", chunks=chunks)
    rename = {old: new for old, new in ALIASES.items() if old in ds.variables or old in ds.dims}
    ds = ds.rename(rename)
    if "mslp" not in ds:
        raise KeyError(f"No MSLP variable found. Available variables: {list(ds.data_vars)}")
    return ds.mslp.transpose("time", "lat", "lon")


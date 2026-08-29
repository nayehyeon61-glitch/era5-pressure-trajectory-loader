from __future__ import annotations

import argparse

from .detect import DetectionConfig, detect_pressure_centres
from .io import open_era5_mslp
from .track import TrackingConfig, build_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pressure-centre trajectories from ERA5 MSLP")
    parser.add_argument("inputs", nargs="+", help="ERA5 NetCDF files")
    parser.add_argument("--output", default="pressure_trajectories.parquet")
    parser.add_argument("--min-anomaly-hpa", type=float, default=2.0)
    parser.add_argument("--min-separation-km", type=float, default=350.0)
    parser.add_argument("--max-speed-kmh", type=float, default=120.0)
    args = parser.parse_args()

    mslp = open_era5_mslp(args.inputs, chunks={"time": 24})
    centres = detect_pressure_centres(
        mslp,
        DetectionConfig(
            min_anomaly_hpa=args.min_anomaly_hpa,
            min_separation_km=args.min_separation_km,
        ),
    )
    tracks = build_trajectories(centres, TrackingConfig(max_speed_kmh=args.max_speed_kmh))
    tracks.to_parquet(args.output, index=False)
    print(f"saved {len(tracks)} points from {tracks.track_id.nunique()} tracks to {args.output}")


if __name__ == "__main__":
    main()


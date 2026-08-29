# ERA5 Pressure Trajectory Loader

ERA5 mean sea-level pressure (MSLP)에서 고기압(H)·저기압(L) 중심을 검출하고, 시간에 따라 `(x=lon, y=lat, pressure)` trajectory로 연결한 뒤 PyTorch 학습 표본으로 반환합니다.

## 파이프라인

1. NetCDF의 MSLP를 `(time, lat, lon)`으로 정규화
2. 국소 최대/최소와 주변 평균 대비 pressure anomaly로 H/L 중심 검출
3. 최소 중심 간 거리로 중복 중심 제거
4. 이동속도 gate + Hungarian assignment로 같은 H/L끼리 연결
5. `history` 구간으로 이후 `horizon` 구간의 위치·압력을 예측

trajectory 한 점은 다음 열을 가집니다.

| 열 | 의미 |
|---|---|
| `time` | ERA5 시각 |
| `type` | `H` 또는 `L` |
| `lat`, `lon` | 중심의 y/x 좌표 |
| `pressure_hpa` | 중심 MSLP |
| `anomaly_hpa` | 주변 평균 대비 편차 |
| `track_id`, `step` | trajectory 식별자와 순서 |

## 설치 및 실행

```bash
pip install -e '.[io,test]'
pytest

era5-build-trajectories raw/2025_*.nc \
  --output pressure_trajectories.parquet \
  --min-anomaly-hpa 2.0 \
  --min-separation-km 350 \
  --max-speed-kmh 120
```

```python
import pandas as pd
from torch.utils.data import DataLoader
from era5_pressure import PressureTrajectoryDataset

tracks = pd.read_parquet("pressure_trajectories.parquet")
dataset = PressureTrajectoryDataset(tracks, history=24, horizon=12)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

batch = next(iter(loader))
# batch["history"]: [B, 24, 5]
# batch["target"]:  [B, 12, 5]
# feature = [lat, lon, pressure_hPa, anomaly_hPa, elapsed_hours]
```

## 6개월 학습 / 3개월 평가

trajectory를 먼저 전체 9개월에서 만들고, 중심 시각을 기준으로 분리합니다. trajectory 연결 직후 split해야 경계 부근의 이동 정보가 보존됩니다. 단, 학습 window가 평가 기간으로 넘어가지 않게 잘라야 data leakage를 피할 수 있습니다.

```python
cutoff = pd.Timestamp("2025-07-01")
train_tracks = tracks[tracks.time < cutoff]
test_tracks = tracks[tracks.time >= cutoff]

train_set = PressureTrajectoryDataset(train_tracks, history=24, horizon=12)
test_set = PressureTrajectoryDataset(test_tracks, history=24, horizon=12)
```

## 설계 범위

현재 중심 추적의 핵심 입력은 MSLP입니다. 이후 `Z500`, `u/v850`, `T850`을 중심 주변에서 sampling해 trajectory의 부가 feature로 붙이는 구조로 확장할 수 있습니다. 열대저기압처럼 폐곡선·와도 조건이 중요한 경우에는 MSLP 국소 최소만으로 분류하지 말고 850 hPa 상대와도와 warm-core 조건을 추가해야 합니다.

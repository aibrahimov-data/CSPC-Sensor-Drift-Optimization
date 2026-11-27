# Drift and sensor stability fact sheet

Files: `src/drift.py`, `results/tables/sensor_stability.csv`,
`results/tables/sensor_drift_by_batch.csv`,
`results/figures/sensor_drift_heatmap.png`.

## Method

- Standardized mean shift for one feature, gas and batch:
  (batch mean - reference mean) / reference standard deviation. The
  reference is the initial training period (batches 1-3), computed separately
  for each gas.
- A sensor-level shift is the mean of the absolute shifts of its eight
  features (f001-f008 = sensor 1, ..., f121-f128 = sensor 16).
- Only observations inside the common concentration range 10-300 ppmv are
  used: 5,431 of 5,491 observations (60 removed). Concentration is never a
  model input.
- **Stability score (used for sensor selection, source-only):** the pooled
  validation batches (4-5) are compared with batches 1-3, and the sensor-level
  shift is averaged over the two gases. A lower score means a more stable
  sensor; rank 1 is the most stable. Batches 6-10 are not used.
- **Drift heatmap (descriptive only):** the same shift for all batches 1-10.
  The values for batches 6-10 are not used for sensor selection.

## Stability ranking (source-only)

| Rank | Sensor | Stability score |
|---:|---:|---:|
| 1 | 6 | 0.509 |
| 2 | 5 | 0.541 |
| 3 | 4 | 0.627 |
| 4 | 10 | 0.631 |
| 5 | 9 | 0.633 |
| 6 | 3 | 0.660 |
| 7 | 2 | 0.757 |
| 8 | 1 | 0.773 |
| 9 | 12 | 0.810 |
| 10 | 11 | 0.842 |
| 11 | 14 | 0.874 |
| 12 | 7 | 0.920 |
| 13 | 13 | 0.952 |
| 14 | 16 | 0.986 |
| 15 | 15 | 1.019 |
| 16 | 8 | 1.051 |

- Scores range from 0.51 to 1.05. Ranks 3-5 (sensors 4, 10, 9) differ by less
  than 0.01, so the boundary of a 4-sensor subset (rank 4 versus rank 5)
  depends on a very small difference. The gap between rank 8 (sensor 1,
  0.773) and rank 9 (sensor 12, 0.810) is larger.

## Observations from the heatmap

- Batches 1-3 are the reference period, so their values are low or moderate
  by construction and are not evidence of stability.
- Ethanol: several sensors (for example 1, 2, 7, 8, 13, 15, 16) show large
  shifts in batch 4, and sensors 1, 2, 9 and 10 show large shifts in batches
  8-9.
- Ethylene: sensors 7, 8 and 11-16 show large shifts in batches 8-9, and
  sensors 1, 2, 9 and 10 show large shifts in batch 10.
- The pattern differs between sensors, gases and batches. No sensor is
  uniformly stable across all batches and both gases, and the shift does not
  increase steadily with batch number. This agrees with the fact that batch
  IDs are not equally spaced in time.
- The colour scale is capped at the 95th percentile (about 1.7), so the
  brightest cells indicate "at or above this value".

## Limitations

- Larger shift does not prove sensor ageing. Concentration distributions,
  experimental conditions and the strength of class separation can also
  change feature means. The 10-300 ppmv filter removes only 60 observations
  and does not make the concentration distributions equal.
- The stability score uses only batches 4-5, which contain few observations
  per gas, so the ranking is noisy. Pooling batches 4 and 5 hides differences
  between them.
- Shifts are measured against batches 1-3, which themselves differ from each
  other; absolute values should be read relative to each other, not as "no
  drift" versus "drift".
- Heatmap values for batches 6-10 are descriptive. They were not used to
  define or change the stability ranking.
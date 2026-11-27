# Sensor selection fact sheet (source-only)

Files: `src/selection.py`, `results/tables/sensor_rankings.csv`,
`results/tables/selected_sensors.csv`,
`results/figures/sensor_importance_vs_stability.png`.

All rankings use the source batches only (training 1-3, validation 4-5).
Batches 6-10 are not used.

## Method

- **Importance:** KNN (k = 21, all 16 sensors) is fitted on batches 1-3. On
  the validation batches (4-5) the eight features of one sensor are shuffled
  together across observations (20 repetitions, seed 42) and the increase of
  the Brier score is recorded. A larger increase means a more important
  sensor. The Brier score is used because the validation balanced accuracy is
  already 1.0 and cannot separate the sensors.
- **Stability:** see `docs/drift_fact_sheet.md` (batches 4-5 compared with
  batches 1-3, lower = more stable).
- **Hybrid:** mean of the importance rank and the stability rank (50% / 50%).
  Ties go to the more important sensor.
- **Random:** for each sampling seed (11, 22, 33, 44, 55) a different set of
  unique sensors is drawn with that seed.
- **Budgets:** 4 sensors (32 features) and 8 sensors (64 features) for random,
  importance, stability and hybrid selection. Budget 16 is the full array
  ("all"). Every selected sensor keeps all eight features.

## Sensor rankings (sorted by hybrid rank)

| Sensor | Importance (mean ± std) | Imp. rank | Stability score | Stab. rank | Hybrid rank |
|---:|---:|---:|---:|---:|---:|
| 9 | 0.0383 ± 0.0128 | 1 | 0.633 | 5 | 1 |
| 5 | 0.0002 ± 0.0004 | 5 | 0.541 | 2 | 2 |
| 6 | -0.0002 ± 0.0002 | 6 | 0.509 | 1 | 3 |
| 10 | 0.0139 ± 0.0064 | 4 | 0.631 | 4 | 4 |
| 1 | 0.0194 ± 0.0092 | 2 | 0.773 | 8 | 5 |
| 2 | 0.0144 ± 0.0045 | 3 | 0.757 | 7 | 6 |
| 4 | -0.0003 ± 0.0006 | 7 | 0.627 | 3 | 7 |
| 3 | -0.0005 ± 0.0007 | 8 | 0.660 | 6 | 8 |
| 14 | -0.0006 ± 0.0005 | 10 | 0.874 | 11 | 9 |
| 13 | -0.0006 ± 0.0005 | 9 | 0.952 | 13 | 10 |
| 12 | -0.0015 ± 0.0004 | 13 | 0.810 | 9 | 11 |
| 11 | -0.0025 ± 0.0009 | 14 | 0.842 | 10 | 12 |
| 15 | -0.0007 ± 0.0005 | 11 | 1.019 | 15 | 13 |
| 16 | -0.0009 ± 0.0005 | 12 | 0.986 | 14 | 14 |
| 7 | -0.0031 ± 0.0006 | 15 | 0.920 | 12 | 15 |
| 8 | -0.0035 ± 0.0008 | 16 | 1.051 | 16 | 16 |

## Selected sensor subsets

| Strategy | Budget | Sensors |
|---|---:|---|
| importance | 4 | 1, 2, 9, 10 |
| stability | 4 | 4, 5, 6, 10 |
| hybrid | 4 | 5, 6, 9, 10 |
| importance | 8 | 1, 2, 3, 4, 5, 6, 9, 10 |
| stability | 8 | 1, 2, 3, 4, 5, 6, 9, 10 |
| hybrid | 8 | 1, 2, 3, 4, 5, 6, 9, 10 |

Random subsets (seed: sensors):

| Seed | 4 sensors | 8 sensors |
|---:|---|---|
| 11 | 2, 8, 12, 14 | 1, 2, 6, 8, 9, 10, 11, 14 |
| 22 | 4, 6, 10, 11 | 2, 3, 4, 7, 8, 11, 13, 15 |
| 33 | 6, 7, 10, 12 | 5, 7, 8, 11, 13, 14, 15, 16 |
| 44 | 2, 5, 9, 13 | 2, 4, 6, 7, 9, 10, 15, 16 |
| 55 | 11, 12, 13, 14 | 3, 4, 9, 10, 11, 12, 15, 16 |

## Observations

- Importance is concentrated in four sensors (9, 1, 2 and 10), whose mean
  Brier increase (0.014-0.038) is clearly larger than its spread over the
  repetitions. For the other twelve sensors the importance is close to zero
  or slightly negative (between about -0.0035 and +0.0002), so their
  importance ranks (5-16) rest on very small differences.
- Importance and stability disagree. The 4-sensor importance subset
  (1, 2, 9, 10) and stability subset (4, 5, 6, 10) share only sensor 10. The
  hybrid subset (5, 6, 9, 10) combines the two.
- At 8 sensors the importance, stability and hybrid rankings select the same
  subset (sensors 1, 2, 3, 4, 5, 6, 9, 10). These three strategies therefore
  cannot be distinguished at this budget; they differ only at 4 sensors.

## Limitations

- Importance is measured on a small validation set (175 observations) with a
  single fitted model (KNN, k = 21) and one shuffling seed.
- Permutation importance can underestimate sensors whose information is also
  carried by other sensors, so a low importance does not mean that a sensor
  contains no information.
- The stability score uses only batches 4-5 and is noisy (see
  `docs/drift_fact_sheet.md`).
- The importance ranking is computed once and is used for both classifiers.
- The rankings describe the source period only. They do not show that the
  selected sensors perform well on future batches.
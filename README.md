# CSPC-Sensor-Drift-Optimization
Sensor selection and recalibration optimization for ethanol-ethylene classification under temporal sensor drift.

Input: all commands below read `data/processed/ethanol_ethylene.csv`, which
is created from the raw UCI files by `src/parse.py`. Tests: `python -m pytest -q`.

### KNN classifier (`src/knn.py`, `src/knn_report.py`)

Run from the repository root: `python -m src.knn` and `python -m src.knn_report`

- `StandardScaler` + `KNeighborsClassifier` on the 128 sensor features
  (concentration is not a model input). The scaler is fitted on training
  data only.
- k is selected on the validation batches (4–5) from the values in
  `config.yaml`: highest validation Balanced Accuracy, ties resolved towards
  the smallest k. Selected k = 21.
- Outputs: `results/tables/knn_validation.csv`,
  `results/tables/knn_validation_recall.csv`,
  `results/figures/knn_k_curve.png`, `docs/knn_fact_sheet.md`.

### PCA (`src/pca_analysis.py`)

Run from the repository root: `python -m src.pca_analysis`

- Scaler and PCA are fitted on source batches 1–5 only and applied to
  batches 6–10. The result is a descriptive visualisation, not a
  classification result.
- Outputs: `results/tables/pca_explained_variance.csv`,
  `results/figures/pca_by_gas.png`, `results/figures/pca_by_batch.png`,
  `docs/pca_fact_sheet.md`.

### Sensor-to-feature mapping (`src/sensors.py`)

Sensor s (1–16) uses the eight consecutive features f(8(s−1)+1) … f(8s),
e.g. sensor 1 = f001–f008, sensor 2 = f009–f016.

### Drift and sensor stability (`src/drift.py`)

Run from the repository root: `python -m src.drift`

- Standardized mean shift per feature, gas and batch: (batch mean − mean of
  batches 1–3) / std of batches 1–3. A sensor's shift is the mean absolute
  shift of its eight features. Only observations within 10–300 ppmv are used.
- Stability score (used for sensor selection): pooled batches 4–5 compared
  with batches 1–3, averaged over both gases. Lower = more stable. Batches
  6–10 are never used for the ranking.
- Outputs: `results/tables/sensor_stability.csv` (selection),
  `results/tables/sensor_drift_by_batch.csv` and
  `results/figures/sensor_drift_heatmap.png` (descriptive only),
  `docs/drift_fact_sheet.md`.

### Sensor selection (`src/selection.py`)

Run from the repository root: `python -m src.selection`

- Strategies: all, random, importance, stability, hybrid, for 4 and 8 sensors
  (budget 16 = all sensors). All rankings use source batches 1–5 only.
- Importance: KNN (selected k) fitted on batches 1–3; the eight features of one
  sensor are shuffled together on batches 4–5 and the increase of the Brier
  score is measured.
- Hybrid: mean of the importance rank and the stability rank.
- Random: one subset per sampling seed (11, 22, 33, 44, 55).
- Outputs: `results/tables/sensor_rankings.csv`,
  `results/tables/selected_sensors.csv` (one row per subset, `features`
  separated by `;`), `results/figures/sensor_importance_vs_stability.png`,
  `docs/selection_fact_sheet.md`.
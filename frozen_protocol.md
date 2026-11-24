# CSPC Sensor Drift Optimization
## Frozen Research Protocol

### 1. Research Question

Can a smaller, drift-resistant sensor array maintain reliable
ethanol-ethylene classification while requiring fewer labeled
recalibration measurements?

### 2. Dataset

Source: UCI Gas Sensor Array Drift at Different Concentrations

Dataset ID: 270

Expected dataset properties:
- 13,910 total observations
- 6 gas classes
- 16 chemical sensors
- 8 features per sensor
- 128 total features
- 10 temporal batches

Selected classes:
- Ethanol: class 1
- Ethylene: class 2

Expected binary subset: 3,577 observations.

All expected counts must be verified against the raw files.

### 3. Temporal Data Splitting

Initial training:
- Batches 1, 2, 3

Validation:
- Batches 4, 5

Final source training:
- Batches 1, 2, 3, 4, 5

Future evaluation:
- Batches 6, 7, 8, 9, 10

No future batch may be used during initial model selection.

### 4. Classification Models

Baseline:
- Dummy Classifier

Main classifier:
- Logistic Regression

Comparison classifier:
- Random Forest

Model selection is performed using source data only.

### 5. Sensor Selection

We compare the following sensor budgets:

- 4 sensors (32 features)
- 8 sensors (64 features)
- 16 sensors (128 features)

Sensor selection strategies:

1. All sensors
2. Random sensor selection
3. Importance-based selection
4. Stability-based selection
5. Hybrid selection

Sensor rankings must be determined without using
future evaluation labels.

### 6. Recalibration Experiment

Target batches:
- Batch 7
- Batch 10

For each target batch:

1. Reserve 30% of each gas class as a locked test set.
2. Use the remaining 70% as calibration candidates.
3. Select equal labeled samples from both gas classes.
4. Retrain the model with source data and calibration samples.
5. Evaluate only on the locked test set.

Calibration budgets per gas:

- 0 samples
- 5 samples
- 10 samples
- 20 samples

Sampling seeds:
- 11
- 22
- 33
- 44
- 55

The locked holdout remains identical across all experiments.

### 7. Evaluation Metrics

Primary:
- Balanced Accuracy
- Macro F1

Secondary:
- Recall for each gas
- Confusion Matrix

Report results separately for each target batch.

### 8. Concentration Analysis

The common nominal concentration range is 10–300 ppmv.

We will:
- Compare concentration distributions.
- Evaluate a concentration-only diagnostic baseline.
- Investigate concentration-matched subsets where feasible.
- Clearly report remaining confounding limitations.

### 9. Data Leakage Prevention

The following rules are mandatory:

- Never fit a scaler on future test observations.
- Never select sensors using target test labels.
- Never tune hyperparameters on final test results.
- Never train on locked recalibration holdout observations.
- Apply identical test sets across competing methods.

### 10. Main Scientific Output

Sensor-Calibration Frontier

We investigate how classification performance depends on:

1. Number of sensors.
2. Number of labeled recalibration measurements.
3. Sensor selection strategy.

The main objective is to identify practical trade-offs
between sensor count, calibration effort, and performance.

### 11. Scientific Limitations

Temporal performance changes do not automatically prove
physical sensor ageing.

Batch numbers are not equally spaced measurement times.

The proposed sensor-calibration comparison is not claimed
to be a completely new algorithm.
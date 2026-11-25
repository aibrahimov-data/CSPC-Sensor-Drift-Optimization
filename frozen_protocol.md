# CSPC Sensor Drift Optimization
## Frozen Research Protocol

### 1. Research Question

**Can a smaller, drift-resistant sensor array maintain reliable ethanol–ethylene classification while requiring fewer labeled recalibration measurements?**

The project investigates whether reducing the number of chemical sensors, combined with limited recalibration, can maintain reliable gas classification under temporal sensor drift.

The main objectives are:

1. Evaluate the effect of temporal sensor drift on classification performance.
2. Compare K-Nearest Neighbors (KNN) and Decision Tree classifiers.
3. Identify sensor subsets that provide reliable classification.
4. Determine how much labeled recalibration data is needed to maintain performance.
5. Investigate trade-offs between sensor count, calibration effort, and classification performance.

### 2. Dataset

**Source:** UCI Gas Sensor Array Drift at Different Concentrations

**Dataset ID:** 270

Dataset properties:

- Total observations: **13,910**
- Gas classes: 6
- Chemical sensors: 16
- Features per sensor: 8
- Total sensor features: 128
- Temporal batches: 10

Selected classes:

- Ethanol: Class 1 — **2,565 observations**
- Ethylene: Class 2 — **2,926 observations**

**Binary dataset: 5,491 observations**

These counts have been verified using the raw UCI dataset.

Each observation contains:

- `sample_id`
- `batch_id`
- `gas_id`
- `concentration_ppmv`
- `f001` through `f128`

The original sensor measurements must be preserved during parsing.

### 3. Temporal Data Splitting

The dataset is divided chronologically to simulate classification on future sensor measurements.

**Initial Training**

- Batches: 1, 2, 3
- Observations: 1,541

**Validation**

- Batches: 4, 5
- Observations: 175

**Final Source Training**

- Batches: 1, 2, 3, 4, 5
- Observations: 1,716

**Future Evaluation**

- Batches: 6, 7, 8, 9, 10
- Observations: 3,775

The initial training and validation datasets are used for model selection and hyperparameter tuning.

After selecting model parameters, the final source models are trained using Batches 1–5.

Future batches must not be used for initial model selection or hyperparameter tuning.

For recalibration experiments, selected future batches are divided into calibration candidates and locked test observations according to Section 6.

### 4. Classification Models

The project uses classification methods covered in CSPC Lecture 4.

**Baseline Model**

- Dummy Classifier (`most_frequent`)

**Main Classification Models**

1. K-Nearest Neighbors (KNN)
2. Decision Tree Classifier

#### 4.1. Dummy Classifier

The Dummy Classifier provides a simple baseline by predicting the most frequent class in the training dataset.

It does not learn relationships between sensor features and gas classes.

Its performance provides a reference for evaluating the main classifiers.

#### 4.2. K-Nearest Neighbors

KNN classifies observations based on the classes of nearby training observations.

Implementation requirements:

- Use `KNeighborsClassifier`.
- Apply `StandardScaler`.
- Fit the scaler using training data only.
- Select the number of neighbors using source validation data.
- Evaluate Balanced Accuracy and Macro F1.
- Investigate sensitivity to sensor selection and recalibration.

**Selection of k (defined before any evaluation of future batches):**

- Candidate values: k ∈ {1, 3, 5, 7, 9, 11, 15, 21, 31, 41}, as listed in `config.yaml`.
- Each candidate is fitted on the initial training batches (1–3) and scored on the validation batches (4–5).
- The selected k is the value with the highest validation Balanced Accuracy. Ties are resolved in favour of the smallest k.
- k is selected once using all 16 sensors. The same k is used for every sensor subset (4, 8, 16 sensors) and every recalibration budget.
- The final KNN models are refitted on batches 1–5 (plus calibration samples in the recalibration experiment) using the selected k.
- Result: k = 21. Validation Balanced Accuracy is 1.000 for k = 21, 31 and 41, so the smallest value is selected.
- k is not changed after inspecting results on batches 6–10.

#### 4.3. Decision Tree

Decision Tree performs classification through successive feature-based decisions.

Implementation requirements:

- Use `DecisionTreeClassifier`.
- Use entropy as the primary splitting criterion.
- Compare different tree depths.
- Select hyperparameters using source validation data.
- Compare training and validation performance.
- Investigate overfitting.
- Evaluate sensitivity to sensor selection and recalibration.

Both main classifiers must use identical dataset splits and evaluation metrics.

### 5. Sensor Selection

We compare three sensor budgets:

| Sensor Budget | Number of Features |
|---|---:|
| 4 sensors | 32 |
| 8 sensors | 64 |
| 16 sensors | 128 |

Each sensor contributes eight features.

The following selection strategies will be investigated:

1. **All Sensors:** Use all 16 sensors as the full-array reference.
2. **Random Selection:** Select sensor subsets using reproducible random seeds.
3. **Importance-Based Selection:** Rank sensors according to their predictive contribution.
4. **Stability-Based Selection:** Rank sensors according to their stability across source batches.
5. **Hybrid Selection:** Combine predictive importance and source-period stability.

Sensor selection methods must use only information available from the permitted source training and validation periods.

Final evaluation batches must not influence the selection of sensor subsets.

For each selected sensor, all eight associated features are retained.

Selection methods and their parameters must be defined before inspecting final test performance.

### 6. Recalibration Experiment

**Target batches:**

- Batch 7
- Batch 10

These batches contain sufficient observations for the planned calibration experiments.

For each target batch:

1. Separate observations by gas class.
2. Reserve approximately 30% as a locked test set using stratified sampling.
3. Use the remaining approximately 70% as calibration candidates.
4. Select equal numbers of labeled calibration observations from each gas class.
5. Combine selected calibration observations with the source training dataset.
6. Retrain the classifier using the selected calibration budget.
7. Evaluate the model only on the locked test set.
8. Repeat the experiment using predefined random seeds.

**Calibration budgets per gas:**

- 0 samples
- 5 samples
- 10 samples
- 20 samples

Therefore, the total number of new labeled measurements is:

| Samples per Gas | Total Calibration Samples |
|---|---:|
| 0 | 0 |
| 5 | 10 |
| 10 | 20 |
| 20 | 40 |

**Sampling seeds:**

- 11
- 22
- 33
- 44
- 55

**Locked holdout seed:** 2026

The locked test set must remain identical across all calibration budgets, sensor selection strategies, and classifiers for the same target batch.

Calibration candidates must never overlap with locked test observations.

The zero-calibration baseline must be evaluated on the same locked test observations as the recalibrated models.

Batches 6, 8, and 9 remain available for evaluation of the original source-trained models without target-batch recalibration.

### 7. Evaluation Metrics

**Primary metrics:**

- Balanced Accuracy
- Macro F1

**Secondary metrics:**

- Accuracy
- Precision
- Recall for each gas class
- Confusion Matrix

The evaluation must report:

1. Source validation performance.
2. Future batch classification performance.
3. Performance under different sensor budgets.
4. Performance under different recalibration budgets.
5. Differences between KNN and Decision Tree.
6. Variation across calibration sampling seeds.

Results must be reported separately for each target batch where applicable.

### 8. Concentration Sensitivity Analysis

The nominal common concentration range is:

**10–300 ppmv**

Different concentration distributions may influence classification performance.

Therefore, the project will:

1. Compare concentration distributions for Ethanol and Ethylene.
2. Repeat selected classification experiments within the common concentration range.
3. Investigate concentration-matched subsets where feasible.
4. Evaluate a concentration-only diagnostic classifier.
5. Compare concentration-controlled results with the main experiments.
6. Discuss remaining concentration-related confounding.

The main KNN and Decision Tree classifiers must use sensor features only.

Concentration is retained for diagnostic analysis and must not be used as an input feature in the primary classification experiments.

### 9. Data Leakage Prevention

The following rules are mandatory:

- Never fit a scaler using future test observations.
- Never select sensors using final test labels.
- Never use final test results for hyperparameter tuning.
- Never train on locked recalibration test observations.
- Never use locked test observations to calculate calibration statistics.
- Never use future batch information to determine source-period sensor stability rankings.
- Use identical evaluation observations for competing models.
- Keep the initial source-only evaluation separate from the recalibration experiments.

All preprocessing, classification, and sensor selection procedures must follow the predefined temporal evaluation protocol.

### 10. Main Scientific Output

## Sensor–Calibration Frontier

The main scientific result will investigate how classification performance depends on:

1. Number of sensors.
2. Number of labeled recalibration measurements.
3. Sensor selection strategy.
4. Classification model.

The project will construct a **Sensor–Calibration Performance Frontier**.

The analysis will identify combinations of sensor count and calibration effort that achieve favorable classification performance.

A Pareto analysis may be used to identify combinations for which neither sensor count nor calibration effort can be reduced without sacrificing another objective.

The project will compare these trade-offs for KNN and Decision Tree.

The objective is not necessarily to maximize classification accuracy at any cost, but to investigate whether reliable classification can be achieved using fewer resources.

### 11. Scientific Limitations

Temporal changes in classification performance do not automatically prove physical sensor ageing.

Other factors, including experimental conditions and concentration distributions, may contribute to observed differences.

Batch numbers are not equally spaced time intervals.

Sensor selection results may depend on the selected source batches and classification model.

Recalibration results may differ between target batches.

PCA projections are descriptive visualizations and do not independently demonstrate that two gas classes have become indistinguishable.

The proposed sensor-selection and recalibration comparison is not claimed to constitute a completely new Machine Learning algorithm.

Scientific originality will be assessed through the specific experimental design, combined sensor–calibration analysis, concentration controls, and comparison with previous research.

### 12. Reproducibility Requirements

The project must include:

- Conda environment specification.
- Version-controlled Python source code.
- Frozen experimental configuration.
- Snakemake pipeline.
- Pytest tests.
- Generated result tables.
- Scientific figures.
- README documentation.

The final pipeline must be reproducible from a clean repository checkout.

All random seeds and experimental parameters must be documented.

### 13. Protocol Revision Record

This protocol was revised before final model evaluation to correct:

- The binary dataset count from 3,577 to 5,491 observations.
- Ethanol and Ethylene class counts using verified raw data.
- The principal classification models to KNN and Decision Tree, matching CSPC Lecture 4.

The original research question, temporal evaluation strategy, sensor budgets, and recalibration objectives remain unchanged.

- The KNN selection rule for k (Section 4.2) was added before evaluation of future batches.
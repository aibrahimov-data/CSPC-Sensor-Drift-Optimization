# CSPC Data Contract

## 1. Dataset Source

**Dataset:** UCI Gas Sensor Array Drift at Different Concentrations

**Dataset ID:** 270

**Source:** UCI Machine Learning Repository

The dataset contains measurements from 16 chemical sensors exposed to six different gases.

The measurements are organized into 10 temporal batches.

Each observation contains 128 sensor features.

## 2. Responsibilities

**Initial Parser Implementation:** Ali

**Data Validation and Maintenance:** Ismayil

**Technical Review:** Ali

**Downstream Data Users:**

- Nurlan — KNN, PCA, Sensor Selection, Drift Analysis.
- Ismayil — Decision Tree, Model Evaluation.
- Ali — Temporal Splitting, Recalibration, Optimization, Integration.

All developers must use the same processed data structure.

## 3. Required Outputs

The parser must generate:

1. `data/processed/all_measurements.csv`
2. `data/processed/ethanol_ethylene.csv`
3. `data/processed/data_audit.csv`

The files must be reproducible from the original UCI dataset.

## 4. CSV Schema

Every processed observation must contain:

| Column | Type | Description |
|---|---|---|
| sample_id | String | Unique observation identifier |
| batch_id | Integer | Temporal batch number (1–10) |
| gas_id | Integer | Gas class identifier (1–6) |
| concentration_ppmv | Float | Gas concentration in ppmv |
| f001–f128 | Float | Chemical sensor features |

The full dataset contains **132 columns**.

The binary classification dataset must contain only:

- `gas_id = 1` — Ethanol
- `gas_id = 2` — Ethylene

No other gas classes are permitted in the binary dataset.

## 5. Feature Mapping

The electronic nose contains 16 chemical sensors.

Each sensor contributes eight features.

| Sensor | Feature Columns |
|---|---|
| Sensor 1 | f001–f008 |
| Sensor 2 | f009–f016 |
| Sensor 3 | f017–f024 |
| Sensor 4 | f025–f032 |
| Sensor 5 | f033–f040 |
| Sensor 6 | f041–f048 |
| Sensor 7 | f049–f056 |
| Sensor 8 | f057–f064 |
| Sensor 9 | f065–f072 |
| Sensor 10 | f073–f080 |
| Sensor 11 | f081–f088 |
| Sensor 12 | f089–f096 |
| Sensor 13 | f097–f104 |
| Sensor 14 | f105–f112 |
| Sensor 15 | f113–f120 |
| Sensor 16 | f121–f128 |

Feature ordering must follow the original UCI dataset.

Sensor selection must operate on complete eight-feature sensor groups.

## 6. Verified Dataset Statistics

### Full Dataset

- Total observations: **13,910**
- Gas classes: 6
- Sensor features: 128
- Temporal batches: 10
- Total columns: 132

### Gas Distribution

| Gas ID | Gas | Observations |
|---|---|---:|
| 1 | Ethanol | 2,565 |
| 2 | Ethylene | 2,926 |
| 3 | Ammonia | 1,641 |
| 4 | Acetaldehyde | 1,936 |
| 5 | Acetone | 3,009 |
| 6 | Toluene | 1,833 |
| **Total** | | **13,910** |

### Binary Dataset

- Ethanol: **2,565**
- Ethylene: **2,926**
- Total: **5,491 observations**

**Expected shape:**

`(5491, 132)`

These values were verified from the processed UCI data.

## 7. Binary Dataset Distribution by Batch

| Batch ID | Binary Observations |
|---|---:|
| 1 | 188 |
| 2 | 498 |
| 3 | 855 |
| 4 | 107 |
| 5 | 68 |
| 6 | 1,088 |
| 7 | 1,311 |
| 8 | 60 |
| 9 | 116 |
| 10 | 1,200 |
| **Total** | **5,491** |

These counts must remain unchanged during parsing.

Any filtering required for experimental analysis must occur downstream of the original processed dataset.

## 8. Data Integrity Rules

The parser must:

1. Preserve original batch IDs.
2. Preserve original gas class labels.
3. Preserve concentration values.
4. Preserve all 128 sensor features.
5. Generate stable and unique sample IDs.
6. Verify valid feature indices.
7. Detect duplicate feature indices.
8. Check missing values.
9. Check non-finite numeric values.
10. Detect potential duplicate observations.
11. Preserve raw dataset files without modification.
12. Avoid scaling or normalizing features.
13. Avoid train/test splitting during parsing.
14. Avoid removing observations without documented justification.

Potential duplicate observations must be reported rather than automatically deleted.

## 9. Data Processing Interface

### Primary Input

`data/processed/ethanol_ethylene.csv`

### Feature Columns

`f001` through `f128`

### Target Column

`gas_id`

### Temporal Splitting Column

`batch_id`

### Concentration Analysis Column

`concentration_ppmv`

### Observation Identifier

`sample_id`

The primary Machine Learning classifiers must not use the following columns as predictors:

- `sample_id`
- `batch_id`
- `gas_id`
- `concentration_ppmv`

Only selected sensor feature columns may be used as predictor variables.

## 10. Temporal Splitting Interface

All models must use:

`src.split.make_temporal_splits()`

Expected data splits:

| Split | Batches | Observations |
|---|---|---:|
| Training | 1–3 | 1,541 |
| Validation | 4–5 | 175 |
| Final Source Training | 1–5 | 1,716 |
| Future Evaluation | 6–10 | 3,775 |

The final source training dataset includes the initial training and validation observations.

It must only be used after model hyperparameters have been selected.

Future batches must not be used for initial model selection.

For target-batch recalibration experiments, calibration candidates and locked test sets must be separated before retraining.

## 11. Machine Learning Model Interface

The project compares:

1. K-Nearest Neighbors
2. Decision Tree Classifier

Dummy Classifier provides the baseline.

All models must accept:

- Feature matrix `X`
- Target vector `y`
- Configuration parameters

All classification models must provide:

- `.fit(X_train, y_train)`
- `.predict(X_test)`

Additional evaluation methods may be implemented where necessary.

The KNN implementation must scale features within the training pipeline.

The Decision Tree implementation does not require feature scaling.

All preprocessing transformations must be fitted only on permitted training observations.

## 12. Model Evaluation

All models must be evaluated using the same observations and metrics.

**Primary metrics:**

- Balanced Accuracy
- Macro F1

**Secondary metrics:**

- Accuracy
- Precision
- Recall
- Confusion Matrix

Evaluation results must identify the classifier and corresponding evaluation batch.

The evaluation interface must support use by the recalibration and optimization modules.

## 13. Data Validation

The parser must verify:

- Exactly 13,910 full-dataset observations.
- Exactly 5,491 binary observations.
- Exactly 128 sensor feature columns.
- Exactly 10 temporal batches.
- Valid gas identifiers.
- Valid batch identifiers.
- Unique sample IDs.
- Numeric feature values.
- Valid concentration values.
- Absence of unexpected missing or non-finite values.

Unexpected values must trigger an error or be explicitly documented in the audit report.

## 14. Implementation

### Parsing

File:

`src/parse.py`

Responsible for:

- Reading raw batch files.
- Parsing measurement lines.
- Combining batches.
- Selecting Ethanol and Ethylene.
- Creating data quality reports.
- Saving processed CSV files.

### Temporal Splitting

File:

`src/split.py`

Responsible for:

- Source training and validation splitting.
- Future evaluation splitting.
- Target-batch calibration candidate selection.
- Locked test separation.

### Model Development

Files:

- `src/dummy.py` — Ali
- `src/knn.py` — Nurlan
- `src/decision_tree.py` — Ismayil

### Sensor Analysis

Files:

- `src/pca_analysis.py` — Nurlan
- `src/drift.py` — Nurlan
- `src/selection.py` — Nurlan

### Model Evaluation

File:

`src/evaluate.py`

Responsible:

Ismayil

### Recalibration and Optimization

Files:

- `src/recalibration.py` — Ali
- `src/optimization.py` — Ali

### Pipeline Integration

Files:

- `Snakefile`
- `README.md`
- `config.yaml`

Responsible:

Ali

## 15. Reproducibility Requirements

The project must use:

- Conda environment.
- Python 3.11.
- Version-controlled source code.
- Documented dataset processing.
- Reproducible random seeds.
- Pytest.
- Snakemake.

**Parser execution:**

```bash
python -m src.parse
```

This command must generate all required processed datasets.

The final Snakemake pipeline must reproduce the processed datasets and all reported analytical results from a clean checkout.

## 16. Data Contract Completion Criteria

The data preparation stage is complete when:

1. All 10 raw batch files are processed successfully.
2. The full dataset contains 13,910 observations.
3. The binary dataset contains 5,491 observations.
4. All 128 sensor features are preserved.
5. Sample IDs are unique.
6. Data audit reports are generated.
7. Processed CSV files can be loaded by Pandas.
8. The temporal splitting module accepts the binary dataset.
9. The Dummy Classifier runs successfully on the processed dataset.
10. Nurlan and Ismayil can independently begin their classification models.

The current parser has produced the expected dataset shape and class counts.

Additional automated parser tests and final pipeline reproducibility checks remain part of the project.
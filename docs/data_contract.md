# CSPC Data Contract

## Dataset Source

UCI Gas Sensor Array Drift at Different Concentrations
Dataset ID: 270

## Responsible Developer

Ismayil — Dataset preparation and parsing.

## Required Outputs

1. data/processed/all_measurements.csv
2. data/processed/ethanol_ethylene.csv
3. data/processed/data_audit.csv

## CSV Schema

Every processed observation must contain:

- sample_id: unique string
- batch_id: integer from 1 to 10
- gas_id: integer from 1 to 6
- concentration_ppmv: numeric
- f001 through f128: numeric sensor features

The binary dataset must contain only gas_id 1 and 2.

## Feature Mapping

There are 16 chemical sensors.

Each sensor has 8 consecutive features.

Sensor 1: f001-f008
Sensor 2: f009-f016
Sensor 3: f017-f024
...
Sensor 16: f121-f128

Feature ordering must follow the official UCI description.

## Expected Results

Full dataset:
- 13910 observations
- 6 gas classes

Binary subset:
- Ethanol: 1641 observations
- Ethylene: 1936 observations
- Total: 3577 observations

These counts must be validated against the raw files.

## Data Integrity Rules

1. Preserve original batch IDs.
2. Preserve original gas labels.
3. Preserve concentration information.
4. Preserve all 128 sensor features.
5. Assign a stable unique sample_id to each row.
6. Check for missing and non-finite values.
7. Do not scale sensor features during parsing.
8. Do not perform train/test splitting during parsing.
9. Do not modify raw dataset files.

## Machine Learning Interface

Main input:

data/processed/ethanol_ethylene.csv

Feature columns:

f001 through f128

Target column:

gas_id

Temporal splitting column:

batch_id

Concentration analysis column:

concentration_ppmv

The main classifiers must not use sample_id,
batch_id, gas_id, or concentration_ppmv as
predictor features.

## Validation

The parser must verify:

- Expected observation counts
- Exactly 128 sensor feature columns
- Valid class labels
- Valid batch identifiers
- Unique sample IDs
- Numeric feature values

## Implementation

Parsing code:

src/parse.py

Responsible:

Ismayil

Reviewed by:

Ali
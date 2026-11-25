# KNN fact sheet (source validation only)

- Model: scikit-learn KNeighborsClassifier with StandardScaler, using all 128
  sensor features (16 sensors x 8). Concentration is not used as an input.
- Training: batches 1-3 (1,541 observations). Validation: batches 4-5
  (175 observations: 92 ethanol, 83 ethylene). The scaler is fitted on
  training data only.
- k was chosen from {1, 3, 5, 7, 9, 11, 15, 21, 31, 41} by validation
  balanced accuracy, with ties resolved towards the smallest k (provisional
  rule, not yet frozen). Selected k = 21.
- Results at k = 21: train balanced accuracy 0.991, macro F1 0.993;
  validation balanced accuracy 1.000, macro F1 1.000. Recall is 1.0 for both
  gases; the validation confusion matrix contains no errors.
- Overfitting: the only clear train-validation gap is k = 1 (train 1.000,
  validation 0.918). For k = 21-41 validation (1.000) is above train
  (0.981-0.991), so there is no sign of overfitting. Train accuracy around
  0.99 also gives no sign of underfitting.
- Limitations: the validation set is small (175 observations) and batches 4-5
  are temporally adjacent to the training batches, so a perfect validation
  score does not imply the same performance on batches 6-10. k = 21, 31 and 41
  tie on validation. These numbers say nothing about drift by themselves.
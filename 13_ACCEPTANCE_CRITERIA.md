# 13 — Acceptance Criteria

Project is accepted only when all criteria below are satisfied.

## A. Data

- [ ] `data_ban.xlsx` is read successfully.
- [ ] Sheet `data_ban` is supported.
- [ ] 500 records are detected for the provided dataset.
- [ ] Required columns are validated.
- [ ] Target labels are binary.
- [ ] Missing/duplicate/domain checks exist.
- [ ] `kekerasan_karet` is not used in the production live input vector.

## B. Preprocessing

- [ ] Train/validation/test split is stratified.
- [ ] Split ratio is 70/15/15.
- [ ] Seed is configurable and default is 42.
- [ ] StandardScaler is fit only on train.
- [ ] Feature order is fixed and persisted.

## C. Model

- [ ] PyTorch is used.
- [ ] Model is an MLP.
- [ ] ReLU is used.
- [ ] Output is one binary-class logit.
- [ ] BCEWithLogitsLoss is used.
- [ ] Adam is used.
- [ ] Early stopping is implemented.
- [ ] Best model checkpoint is persisted.

## D. Evaluation

- [ ] Test set is held out until final evaluation.
- [ ] Accuracy is calculated.
- [ ] Precision class 1 is calculated.
- [ ] Recall class 1 is calculated.
- [ ] F1 class 1 is calculated.
- [ ] ROC-AUC is calculated when both test classes are present.
- [ ] Confusion matrix is generated.
- [ ] Threshold is selected without using the test set.

## E. Artifacts

- [ ] `model.pt` exists.
- [ ] `model_config.json` exists.
- [ ] `scaler.json` exists.
- [ ] `hasil_evaluasi.json` exists.
- [ ] `training_history.json` exists.
- [ ] Artifacts can be reloaded in a fresh process.

## F. API

- [ ] `/health` works.
- [ ] `/api/model-info` works.
- [ ] `/api/predict` works.
- [ ] Request validation works.
- [ ] Model-unavailable state returns a safe error.
- [ ] API uses persisted model/scaler rather than a new random model.

## G. Dashboard

- [ ] Dashboard loads.
- [ ] Four production inputs are visible.
- [ ] Ideal ranges are shown.
- [ ] Prediction is requested from API.
- [ ] Prediction result is understandable.
- [ ] Good vs rework probability is visible.
- [ ] Evaluation metrics are visible.
- [ ] Confusion matrix is visible.
- [ ] Training history is visible.
- [ ] Artifact/model information is visible.
- [ ] API error/loading states exist.
- [ ] Hardness is not required for live prediction.

## H. Testing

- [ ] Unit tests pass.
- [ ] API integration tests pass.
- [ ] Artifact reload test passes.
- [ ] Prediction output is finite.
- [ ] Probability is in [0, 1].
- [ ] Boundary input tests execute without server error.

## I. Documentation

- [ ] Root README explains setup.
- [ ] Training command is documented.
- [ ] API run command is documented.
- [ ] Test command is documented.
- [ ] Model feature order is documented.
- [ ] Production vs offline hardness interpretation is documented.

## J. Final verification

Perform a clean end-to-end run:

```text
dataset
  -> train
  -> evaluate
  -> save artifacts
  -> reload artifacts
  -> start API
  -> prediction
  -> dashboard
  -> tests
```

No step may depend on manually editing generated files.

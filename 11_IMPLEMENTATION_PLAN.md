# 11 — Implementation Plan

## Phase 1 — Project bootstrap

Create:

- package structure;
- requirements;
- configuration;
- `.gitignore`;
- base README;
- test scaffolding.

Acceptance:

```text
python -m pytest
```

runs successfully even before ML tests are complete.

## Phase 2 — Data pipeline

Implement:

1. `load_dataset()`
2. `validate_dataset()`
3. `split_dataset()`
4. `fit_preprocessor()`
5. `transform_*()`

Then run against `data_ban.xlsx`.

Output a compact data report.

## Phase 3 — MLP model

Implement:

```python
class BanQualityMLP(nn.Module):
    ...
```

Model architecture must be driven by configuration where practical.

## Phase 4 — Training

Implement:

- deterministic seed;
- DataLoaders;
- BCEWithLogitsLoss;
- Adam;
- early stopping;
- best checkpoint;
- training history.

## Phase 5 — Evaluation

Implement validation threshold selection and final test evaluation.

Do not tune test set.

## Phase 6 — Artifact persistence

Generate all required artifact files.

Then immediately reload them in a separate process or smoke test.

## Phase 7 — API

Implement:

```text
GET  /health
GET  /api/model-info
POST /api/predict
```

Add Pydantic validation.

## Phase 8 — Dashboard

Build UI sections from `07_DASHBOARD_SPEC.md` and `08_UI_DESIGN_SPEC.md`.

Connect only through API. Do not duplicate ML logic in JavaScript.

## Phase 9 — Automated tests

Complete unit/integration tests.

## Phase 10 — Final verification

Fresh run:

```text
delete generated artifacts
-> train
-> verify metrics
-> start API
-> call prediction
-> open dashboard
```

The final build must pass `13_ACCEPTANCE_CRITERIA.md`.

## Suggested commands

Install:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train:

```bash
python -m ml.train
```

Run API:

```bash
uvicorn app.main:app --reload
```

Test:

```bash
python -m pytest -q
```

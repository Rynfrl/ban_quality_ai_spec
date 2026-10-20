# MASTER PROMPT — Build the Ban Quality ML Application

You are the coding agent responsible for implementing the project described by the Markdown specifications in this folder.

## First instruction

Before writing code, read all Markdown files:

```text
README.md
00_AI_BUILD_BRIEF.md
01_REQUIREMENTS.md
02_DATA_SPEC.md
03_ML_SPEC.md
04_TRAINING_SPEC.md
05_MODEL_ARTIFACT_SPEC.md
06_API_SPEC.md
07_DASHBOARD_SPEC.md
08_UI_DESIGN_SPEC.md
09_TESTING_SPEC.md
10_PROJECT_STRUCTURE.md
11_IMPLEMENTATION_PLAN.md
12_AI_AGENT_RULES.md
13_ACCEPTANCE_CRITERIA.md
DATASET_AUDIT.md
```

Treat them as the project contract.

## Core requirement

Build an end-to-end machine learning application for industrial tire quality classification:

```text
data_ban.xlsx
 -> validation
 -> stratified split
 -> StandardScaler
 -> PyTorch MLP + ReLU
 -> training
 -> validation threshold
 -> test evaluation
 -> model/scaler/JSON artifacts
 -> FastAPI inference
 -> web dashboard
```

## Production features

Use exactly this live input vector:

```text
suhu
tekanan
waktu_curing
rasio_karet
```

Target:

```text
0 = Baik
1 = Perlu Perbaikan
```

Do not use `kekerasan_karet` for live production inference.

## Expected deliverable

Produce a working repository matching the recommended project structure and acceptance criteria.

Do not stop at pseudo-code.

Implement actual runnable code, tests, artifact generation, API, and dashboard.

## Required quality bar

The result must:

- train on the provided Excel file;
- produce actual metrics;
- persist the model and preprocessing;
- reload them successfully;
- expose a working prediction API;
- expose a usable dashboard;
- pass automated tests.

## Final report

After implementation, provide:

1. files created/modified;
2. commands to install, train, test, and run;
3. actual final test metrics;
4. artifact locations;
5. any remaining limitations.

Do not fabricate metrics or claim completion if a required step failed.

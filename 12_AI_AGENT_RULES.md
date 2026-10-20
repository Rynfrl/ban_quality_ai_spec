# 12 — AI Coding Agent Rules

## Source of truth

Read all files in `docs/` before making architectural decisions.

## Do

- inspect existing files before editing;
- preserve user data;
- use explicit configuration;
- prefer small testable functions;
- type annotate public interfaces;
- validate inputs;
- log meaningful errors;
- write tests with implementation;
- keep frontend and backend responsibilities separated;
- use the same artifact for evaluation and inference;
- document assumptions.

## Do not

- replace the dataset with synthetic data;
- fabricate model metrics;
- claim accuracy before training;
- use the test set for tuning;
- use `kekerasan_karet` in production live inference;
- hard-code prediction outputs;
- hide model-loading failures;
- silently train a new model on API startup;
- commit secrets;
- return internal stack traces;
- add unnecessary frameworks.

## Important interpretation rule

The case study contains five measured numeric variables in Excel, but the stated process-input table contains four process features. Resolve this by:

- four-feature production model;
- hardness as offline quality/analysis variable.

Do not silently choose a different interpretation.

## Coding quality

Use:

- small functions;
- explicit names;
- docstrings where useful;
- Pydantic schemas;
- pytest;
- deterministic seed;
- pathlib for paths;
- environment/configuration for runtime paths.

Avoid:

- giant single-file applications;
- global mutable state unless encapsulated in a service object;
- duplicated preprocessing;
- duplicated feature definitions.

## Feature single source of truth

Define production feature order in exactly one shared configuration:

```python
FEATURES = (
    "suhu",
    "tekanan",
    "waktu_curing",
    "rasio_karet",
)
```

Backend, training, artifact metadata, tests, and frontend metadata must derive from this source where practical.

## Business interpretation

Model output is a prediction, not a physical certification.

UI copy must avoid guarantees.

## When implementation details are uncertain

Prefer the simplest design satisfying all acceptance criteria. Do not expand scope merely for architectural novelty.

# Dataset Audit Snapshot

This file records the baseline characteristics of the provided dataset so the AI agent knows the starting point before it trains anything.

## Dataset

File: `data_ban.xlsx`

Sheet: `data_ban`

Rows: `500`

Columns:

```text
suhu
tekanan
waktu_curing
rasio_karet
kekerasan_karet
label
```

## Observed statistics

| Column | Min | Max | Mean |
|---|---:|---:|---:|
| suhu | 140.2 | 180.0 | 159.827 |
| tekanan | 15.0 | 25.0 | 19.983 |
| waktu_curing | 8.0 | 15.0 | 11.532 |
| rasio_karet | 0.4 | 0.8 | 0.605 |
| kekerasan_karet | 55.1 | 75.0 | 64.709 |

## Label distribution

```text
0 / Baik            = 325
1 / Perlu Perbaikan = 175
```

Class ratio:

```text
65% / 35%
```

## Data quality observed

- Missing values: 0 across the inspected dataset.
- Complete duplicate rows: 0.
- Labels observed: only 0 and 1.

## Important interpretation

The screenshot/study case describes four **process** inputs:

```text
suhu
tekanan
waktu_curing
rasio_karet
```

The Excel file also includes:

```text
kekerasan_karet
```

Treat hardness as an offline quality/analysis variable for the production model. The production model must not depend on a physical measurement that the proposed workflow is intended to avoid.

## Do not use this file as a substitute for re-running validation

These values are a baseline snapshot. The implemented program must calculate them dynamically from the actual dataset at runtime/training time.

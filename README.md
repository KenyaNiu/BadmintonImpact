# BadmintonImpact

Reproducible code for **context-normalized landing-impact ranking from markerless pose**. The evaluated input is a pre-segmented, impact-aligned badminton landing window; the output is a score used to prioritize high-impact events for coach review.

This repository does not claim continuous-video landing detection, absolute GRF estimation, injury-risk prediction, or autonomous coaching. The exact claim and evaluation lock are in [`docs/RESEARCH_CONTRACT.md`](docs/RESEARCH_CONTRACT.md).

> **Version:** this release corresponds to the corrected canonical `q=0.75` protocol (event-level cohort, learned-attention pooling, one eligibility rule for all models). Results are regenerated from `outputs/runs/corrected_q75/`, which is created by the commands below and is not versioned. Earlier code (`src/`, `tools/`, `training/`) was superseded by this release.

## Repository layout

```text
BadmintonImpact/
├── badminton_impact_ai/ # all Python implementation and the single CLI
├── configs/             # preparation and frozen experiment YAML
├── docs/                # research contract, reproduction guide, data access form
├── paper/figures/       # script that regenerates the result figures
└── tests/               # unit and synthetic end-to-end tests
```

Generated data, checkpoints, predictions, and metrics belong under `outputs/` and are intentionally not versioned.

The Python package has one-way responsibilities:

| Path | Responsibility |
|---|---|
| `cli.py` | The only user-facing command; dispatches four workflow stages. |
| `data/` | NPZ parsing, label construction, cohort rules, splits, and datasets. |
| `models/` | CN-HiLDNet and comparison backbones. |
| `experiment/` | Configuration, training, run provenance, analysis, and paper artifacts. |
| `metrics/` | Per-model classification, regression, calibration, and ranking metrics. |
| `stats/` | Fold-level paired statistical comparisons. |

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,train]"
pytest -q
```

## Reproduction

Copy the preparation template and set `data_root` to the extracted BadmintonGRF Tier-1 directory. The source dataset is read-only.

```bash
cp configs/preparation.example.yaml configs/preparation.local.yaml
badminton-impact prepare --config configs/preparation.local.yaml

badminton-impact run \
  --config configs/experiments/main.yaml \
  --run-dir outputs/runs/corrected_q75 \
  --check-only

badminton-impact run \
  --config configs/experiments/smoke.yaml \
  --run-dir outputs/runs/smoke
badminton-impact analyze --run-dir outputs/runs/smoke
```

Run the full configuration only after the smoke run passes:

```bash
badminton-impact run \
  --config configs/experiments/main.yaml \
  --run-dir outputs/runs/corrected_q75 \
  --resume
badminton-impact analyze --run-dir outputs/runs/corrected_q75
badminton-impact artifacts \
  --run-dir outputs/runs/corrected_q75 \
  --out-dir paper/generated
```

Every experiment directory records the resolved configuration, input hashes, environment, cohort, splits, predictions, metrics, checkpoints, and completion state. See [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) for acceptance gates.

## Data access

The processed BadmintonGRF data products are available through a controlled request process. Complete [`docs/BadmintonImpact_data_access_request_form.md`](docs/BadmintonImpact_data_access_request_form.md) and email it to the address given in the form. Dataset files are not included in this repository.

## License

Code is released under the [MIT License](LICENSE). Dataset files retain the terms of their original release and are not included in this repository.

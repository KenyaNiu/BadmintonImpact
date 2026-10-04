# Reproducing the results

The reported numbers come from one canonical run, `outputs/runs/corrected_q75/`. Everything below reads and writes
only under `outputs/`; the source dataset is never modified.

## 1. Install and test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
pytest -q            # must pass before any data preparation or training
```

## 2. Prepare the data

Copy `configs/preparation.example.yaml` to `configs/preparation.local.yaml`, set `data_root` to the read-only
BadmintonGRF Tier-1 directory (do not commit the local file), then run:

```bash
badminton-impact prepare --config configs/preparation.local.yaml
```

This writes event-weighted fold labels and one statistical-feature row per camera view to `outputs/prepared/`.

## 3. Check, then run a smoke test

```bash
badminton-impact run --config configs/experiments/main.yaml --run-dir outputs/runs/corrected_q75 --check-only
badminton-impact run --config configs/experiments/smoke.yaml --run-dir outputs/runs/smoke
badminton-impact analyze --run-dir outputs/runs/smoke
```

`--check-only` validates the cohort and splits without training. The smoke run is usable when `status.json` says
`complete` and all models share exactly the same test cohort (`analysis/summary.json`).

## 4. Full ten-fold run

Check every gate in [`PROTOCOL.md`](PROTOCOL.md) first. An interrupted run is continued with the
same command plus `--resume` and the identical resolved configuration; per-fold, per-model seeds are derived from
the base seed, fold, model and task mode, so execution order and resuming do not change any initialisation.

```bash
badminton-impact run --config configs/experiments/main.yaml --run-dir outputs/runs/corrected_q75   # add --resume to continue
badminton-impact analyze --run-dir outputs/runs/corrected_q75
badminton-impact artifacts --run-dir outputs/runs/corrected_q75 --out-dir outputs/paper_artifacts
python3 scripts/generate_result_figures.py --run-dir outputs/runs/corrected_q75 --output-dir outputs/figures
```

The result figures are written as editable SVG, vector PDF, EPS and 600-dpi TIFF, each with a source-data CSV.
The figures of the README are produced by `scripts/readme_figures/` (see the docstrings).

## 5. Run-directory contract

```text
config.yaml  status.json  environment.json  input_hashes.json  cohort.csv  splits.csv
checkpoints/  predictions/  metrics/  logs/  analysis/
```

Analysis refuses incomplete runs, and no script reads results outside the selected run directory.

## 6. Scope

The pipeline evaluates ranking of pre-segmented, impact-aligned candidate windows. It does not reproduce
continuous-video landing detection or clinical injury-risk estimation; the locked claim boundary is in
[`PROTOCOL.md`](PROTOCOL.md).

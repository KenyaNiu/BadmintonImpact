# Reproducing the corrected experiments

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,train]"
pytest -q
```

The source checkout must pass all tests before data preparation or training.

The manuscript's numerical results come from the completed canonical run at `outputs/runs/corrected_q75/`. Regenerate the paper-facing artifacts from that directory only.

## 2. Prepare local data

Copy `configs/preparation.example.yaml` to `configs/preparation.local.yaml`, set the read-only BadmintonGRF Tier-1 root, and do not commit the local file.

```bash
badminton-impact prepare --config configs/preparation.local.yaml
```

This produces event-weighted fold labels and exactly one statistical-feature row per camera view under `outputs/prepared/`. It never writes into the source dataset.

## 3. Mandatory smoke run

First run the no-training preflight:

```bash
badminton-impact run \
  --config configs/experiments/main.yaml \
  --run-dir outputs/runs/corrected_q75 \
  --check-only
```

Then run the mandatory one-fold smoke configuration:

```bash
badminton-impact run \
  --config configs/experiments/smoke.yaml \
  --run-dir outputs/runs/smoke
badminton-impact analyze --run-dir outputs/runs/smoke
```

Inspect `status.json`, `cohort.csv`, `splits.csv`, and `analysis/summary.json`. The run is usable only when status is `complete` and all model test cohorts match exactly.

## 4. Corrected main run

Do not launch this command until every gate in `docs/RESEARCH_CONTRACT.md` is checked.

```bash
badminton-impact run \
  --config configs/experiments/main.yaml \
  --run-dir outputs/runs/corrected_q75 \
  --resume
```

Resume an interrupted run only with the identical resolved configuration:

```bash
badminton-impact run \
  --config configs/experiments/main.yaml \
  --run-dir outputs/runs/corrected_q75 \
  --resume
```

Per-fold/model seeds are derived from the base seed, fold, model, and task mode, so execution order and resume do not alter later initializations.

After analysis, generate paper-facing tables from that run only:

```bash
badminton-impact analyze --run-dir outputs/runs/corrected_q75
badminton-impact artifacts \
  --run-dir outputs/runs/corrected_q75 \
  --out-dir paper/generated

python3 paper/figures/generate_result_figures.py \
  --run-dir outputs/runs/corrected_q75 \
  --output-dir paper/figures
```

The three corrected result figures are generated as editable SVGs, LaTeX-ready vector PDFs, journal-upload EPS files, and 600-dpi TIFFs. Each figure also has a source-data CSV beside it. Their visual grammar follows the submitted figures, while every plotted value is regenerated from the corrected canonical run.

## 5. Run-directory contract

Each run contains:

```text
config.yaml
status.json
environment.json
input_hashes.json
cohort.csv
splits.csv
checkpoints/
predictions/
metrics/
logs/
analysis/
```

Analysis refuses incomplete runs. No script reads results outside the selected run directory.

## 6. Scope

The pipeline evaluates ranking on pre-segmented, impact-aligned candidate windows. It does not reproduce continuous-video landing detection or clinical injury-risk estimation. See `docs/RESEARCH_CONTRACT.md` for the locked claim and evaluation boundary.

<h1 align="center">BadmintonImpact</h1>
<p align="center"><b>Context-relative ranking of badminton landing events from markerless pose</b><br>
<sub>Code for the paper in <i>IET Cyber-Systems and Robotics</i> (2026)</sub></p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg"></a>
  <img alt="Tests" src="https://img.shields.io/badge/tests-25%20passing-brightgreen.svg">
  <img alt="Protocol" src="https://img.shields.io/badge/protocol-LOSO%20%C2%B7%20q%3D0.75-informational.svg">
</p>

<p align="center">
  <img src="docs/assets/hero.png" alt="Video frames with 2D pose and force curve of one landing, and the ranked top-20 % review list of the same trial" width="100%">
</p>

<p align="center"><a href="#overview">Overview</a> · <a href="#how-it-works">How it works</a> · <a href="#results">Results</a> · <a href="#quick-start">Quick start</a> · <a href="#repository">Repository</a> · <a href="#data-and-citation">Data & citation</a></p>

---

## Overview

A training centre records many hours of multi-camera badminton video, and a coach cannot inspect every landing. **BadmintonImpact ranks already identified landings by how demanding they are for their own drill and fatigue state**, using only markerless 2D pose, so that a limited review budget (for example the top 20 % of each trial) goes to the most relevant clips.

| | |
|---|---|
| **Input** | A pre-segmented, impact-centred pose window (±0.5 s, COCO-17), 16 window statistics, protocol context (drill, fresh/fatigued) |
| **Output** | One relative-priority score per physical landing (duplicate camera views are merged) |
| **Supervision** | Force-plate peaks, used **only** to build training labels, never as a model input |
| **Evaluation** | Leave-one-subject-out (LOSO), unique-impact AUROC/AUPRC/F1, fixed top-20 % review budget, calibration |

**Not claimed:** continuous-video landing detection, absolute ground-reaction-force estimation, injury-risk assessment, or autonomous coaching. The claim boundary is frozen in [`docs/RESEARCH_CONTRACT.md`](docs/RESEARCH_CONTRACT.md).

---

## How it works

**1. Context-relative labels.** One global force cut-off lets the drill identity predict the label (a smash always looks "high"). Instead, for each LOSO fold *k* and each stage–fatigue context *c*, the threshold uses **training subjects only**:

$$
\tau_{k,c} = Q_{0.75}\left(\{\,p_i : s_i \in \mathcal{S}^{(k)}_{\text{train}},\ \boldsymbol{c}_i = c\,\}\right), \qquad y_i = \mathbb{1}[\,p_i \ge \tau_{k,c_i}\,]
$$

"High impact" therefore means high *for that drill and fatigue state*, and the held-out athlete never touches a threshold. A context-only model scores near chance (AUROC 0.477), so the labels no longer leak the protocol.

<p align="center"><img src="docs/assets/idea.png" alt="A single cut-off labels the drill; a cut-off per context labels relative severity" width="100%"></p>

**2. CN-HiLDNet.** A small task-specific model: a temporal branch (dilated TCN with attention pooling), a window-statistics branch and a protocol-context branch are concatenated and fused, then feed a ranking head and an auxiliary peak-force head. Only the ranking score is used at inference (451,043 parameters).

**3. Review list.** Scores of the camera views of one landing are averaged; within each held-out trial the top 20 % of landings go to the coach.

---

## Results

Ten LOSO folds, 2,242 physical landings (17,267 camera views), one cohort and venue. Unique-impact (one score per landing) is the primary resolution.

| Unique-impact AUROC | CN-HiLDNet | HGB (pose + context) | Paired difference |
|---|---|---|---|
| mean ± SD over folds | **0.942 ± 0.036** | 0.861 ± 0.086 | **0.081** (8.1 percentage points; 95 % CI 0.039–0.132; exact Wilcoxon *p* = 0.00195; positive in all 10 folds) |

<p align="center">
  <img src="docs/assets/results_models.png" alt="Unique-impact AUROC of all models" width="49%">
  <img src="docs/assets/results_top20.png" alt="Precision, recall and NDCG at 20 percent" width="49%">
</p>

Reviewing only the top 20 % of each trial raises precision / recall / NDCG@20 % from 0.464 / 0.517 / 0.712 (HGB) to **0.536 / 0.635 / 0.844**.

<details>
<summary><b>Ablation and limitations</b></summary>

<p align="center"><img src="docs/assets/results_ablation.png" alt="Ablation: AUROC lost when removing components" width="62%"></p>

Temporal encoding and context conditioning are supported. **Pose statistics, learned attention pooling and the auxiliary head show no independent benefit** (their intervals include zero), so the full model is the evaluated implementation, not proof that every part is needed.

- The advantage is concentrated in structured drills; on **rally** footage a distinct benefit is *not* demonstrated (AUROC difference 0.011, 95 % CI −0.142 to 0.219).
- 10 participants, one venue; LOSO cannot establish generalisation across cameras, clubs or protocols.
- One deterministic seed per fit; fold SDs mix participant and training variation.
- Results are for aligned, pre-segmented windows, not an end-to-end video system.

</details>

---

## Quick start

```bash
git clone https://github.com/KenyaNiu/BadmintonImpact.git && cd BadmintonImpact
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"        # Python >= 3.10; install a CUDA/CPU build of torch if needed
pytest -q                            # 25 tests, synthetic data only
```

```bash
cp configs/preparation.example.yaml configs/preparation.local.yaml   # set data_root
badminton-impact prepare --config configs/preparation.local.yaml

badminton-impact run     --config configs/experiments/smoke.yaml --run-dir outputs/runs/smoke   # minutes
badminton-impact analyze --run-dir outputs/runs/smoke

badminton-impact run     --config configs/experiments/main.yaml --run-dir outputs/runs/corrected_q75 --resume
badminton-impact analyze --run-dir outputs/runs/corrected_q75
```

| Command | Purpose |
|---|---|
| `prepare` | Build base labels, context labels and pose features from the BadmintonGRF Tier-1 data |
| `run` | Train and evaluate all models under LOSO (`--check-only` validates, `--resume` continues) |
| `analyze` | Fold metrics, top-K review metrics, calibration, rally subgroups, paired statistics |
| `artifacts` | Write paper-facing tables and numbers |

Every run directory records the resolved configuration, input hashes, environment, cohort, splits, predictions, metrics and checkpoints; acceptance gates are in [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md). The README figures are regenerated by `docs/assets/make_*.py`.

---

## Repository

```text
badminton_impact_ai/   # package and the single CLI
├── data/  models/  experiment/  metrics/  stats/
configs/               # preparation template and frozen experiment YAML
docs/                  # research contract, reproduction guide, data-access form, figures
scripts/               # regenerates the result figures from a completed run
tests/                 # unit and synthetic end-to-end tests
```

---

## Data and citation

The BadmintonGRF data products are shared through a **controlled request process** and are not included here: complete [`docs/BadmintonImpact_data_access_request_form.md`](docs/BadmintonImpact_data_access_request_form.md) and email it to the address in the form. Approved applicants receive the de-identified package needed to reproduce the analyses and may not attempt re-identification or redistribution.

```bibtex
@article{niu2026badmintonimpact,
  title   = {{BadmintonImpact}: Context-Relative Ranking of Pre-Segmented Badminton Landing Events from Markerless Pose for Coach Review},
  author  = {Niu, Kuoye and Song, Xian and Xie, Yilun and Sun, Jia'ao and Yang, Lumeng and Wan, Puyang and Ding, Ziran and Ma, Yong and Li, Jianwei},
  journal = {IET Cyber-Systems and Robotics},
  year    = {2026},
  note    = {Accepted; volume, pages and DOI to be added on publication}
}
```

The code corresponding to the article is tagged [`v1.0-csr-2026`](../../releases/tag/v1.0-csr-2026). Code is released under the [MIT License](LICENSE); dataset files keep the terms of their original release.

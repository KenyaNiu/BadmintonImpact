# Evaluation protocol and claim boundary

The analysis plan below was fixed before the final experiments (2026-08-12). It is an analysis lock, not a
prospective preregistration.

## 1. What is claimed

The model ranks **pre-segmented, impact-aligned candidate landing windows** by how demanding they are *relative to
their own movement and fatigue context*, using markerless pose and protocol metadata. It does **not** evaluate
continuous-video landing detection, raw-video tracking, absolute ground-reaction-force readout, injury-risk
prediction or autonomous coaching.

The force plate is used only to build supervision and to align windows; no force measurement is a model input at
inference. Results therefore characterise the scoring module under oracle-like candidate localisation, not an
end-to-end video system. CN-HiLDNet is an implementation of the formulation; no claim depends on it beating every
simpler baseline.

## 2. Cohort

Every compared model uses exactly the same rows. A row is eligible only if its subject is known, its pose/statistics
record is valid, and both `context_high_impact_q75` and `peak_fz_context_centered` are finite. Unused targets
(impulse, loading rate) never filter the cohort. Before training, `cohort.csv` records fold, role, subject, trial,
camera view, unique-impact key, targets, the eligibility decision and the exclusion reason, and training fails if two
models receive different test keys or labels.

The expected cohort of the full run (17,267 views, 2,242 unique impacts, ten held-out subjects) is a preflight
assertion, not a result to optimise.

## 3. Splits and tuning

- **Outer:** ten leave-one-subject-out folds.
- **Inner validation:** stratified by the binary label and grouped by unique-impact key (no camera view of one impact
  crosses train/validation; both partitions contain both classes).
- Label thresholds, feature normalisation, early stopping, classification thresholds and calibration are fitted
  without the held-out subject. Seed, architecture, epochs, patience, auxiliary-loss weight and checkpoint rule are
  frozen in the resolved run configuration.
- Fold standard deviations summarise fold-specific fits under one initialisation; they are not a pure estimate of
  between-subject variability.

## 4. Outcomes

- **Operational unit:** the unique physical impact (camera views are averaged).
- **Primary endpoint:** fold-wise unique-impact AUROC and the paired CN-HiLDNet-minus-HGB difference with a 95 %
  bootstrap CI.
- **Application endpoint:** within each held-out subject and trial, rank impacts and evaluate the top 20 % review
  budget (Precision@20 %, Recall@20 %, NDCG@20 %); trials without positives are excluded from recall/NDCG and counted.
- **Secondary:** view-level AUROC/AUPRC/F1, calibration, rally/non-rally subgroups, auxiliary peak regression.
- **Statistical unit:** the held-out-subject fold. Wilcoxon signed-rank is exact when no paired difference is zero,
  otherwise the normal approximation is used and the zero count reported.

## 5. Interpretation rules

- A model advantage is claimed only where the paired unique-impact uncertainty excludes zero.
- If only view-level improvement is clear, claim a view-level advantage but not event-level superiority.
- If HGB matches CN-HiLDNet, the supervision and evaluation contributions stand and the simpler model is preferred.
- The quantile, seed, review budget, subgroup or aggregation rule is never chosen after seeing test results.

## 6. Gates before a long run

- [x] cohort manifest and expected counts pass;
- [x] no test-subject information enters labels, transforms, thresholds, calibration or checkpoint selection;
- [x] one config produces one self-contained run directory with config, input hashes, environment and seed;
- [x] a one-fold, one-epoch smoke run completes and all models share identical test keys and labels;
- [x] all unit and synthetic integration tests pass;
- [x] `badminton-impact artifacts` reads only the selected run directory.

# Research contract for the corrected revision

Version 1, frozen on 2026-08-12 before the corrected long-running experiments. This is a revision-time analysis lock, not a claim of prospective preregistration.

## 1. Paper positioning

- **Type:** New Problem/Setting paper with a companion task-specific model.
- **Primary contribution:** a fold-safe, context-relative formulation and evaluation protocol for ranking pre-segmented badminton landing events from markerless pose.
- **Model role:** CN-HiLDNet operationalizes the formulation; the paper does not depend on it outperforming every simpler baseline.

## 2. Claim boundary

The evaluated task starts from pre-segmented, impact-aligned candidate windows. It does not evaluate continuous-video event detection, raw-video tracking, absolute ground-reaction-force readout, injury-risk prediction, or autonomous coaching.

The force plate is used to construct supervision and align Tier-1 windows. No force measurement is a model input at inference. Consequently, results characterize the scoring module under oracle-like candidate localization, not an end-to-end video system.

## 3. Thinking template

| Stage | Locked content |
|---|---|
| Research background | Coaches reviewing many badminton landings need a compact queue of relatively demanding events, while direct force-plate measurement is difficult to scale beyond instrumented sessions. |
| Limitation 1 | Continuous waveform regression and absolute-force thresholds do not directly answer which candidate landings are unusually demanding within a movement and fatigue context. |
| Limitation 2 | Pooled thresholds let protocol metadata predict labels without kinematic evidence, creating a supervision-induced shortcut. |
| Limitation 3 | Camera views of one physical landing are correlated; view-only evaluation can overstate coach-facing evidence and does not test held-out-athlete generalization. |
| Our Goal | Rank pre-segmented landing events by context-relative impact using pose and protocol metadata while excluding the held-out subject from every learned threshold and evaluation decision. |
| Challenge 1 | Contexts have different force distributions, so one pooled boundary confounds movement identity with relative severity. |
| Challenge 2 | Pose sequences, window-level kinematic summaries, and protocol context contain complementary signals but permit shortcut reliance if evaluated carelessly. |
| Challenge 3 | Subject exclusion, duplicate camera views, calibration, and a finite review budget require distinct fold-safe and event-level analyses. |
| Methodology topic sentence | BadmintonImpact combines fold-safe context-normalized supervision, candidate-window scoring, and subject-exclusive event-aware evaluation. |
| Module A | Construct train-subject-only stage–fatigue quantile labels and document every fallback without test-subject statistics. |
| Module B | Compare CN-HiLDNet with cohort-matched HGB and temporal baselines on one immutable eligible cohort; do not call HGB feature-matched because CN additionally uses the raw pose sequence. |
| Module C | Evaluate held-out subjects at unique-impact level, with view-level diagnostics, calibration, subgroup analysis, and a locked trial-level review-budget metric. |
| Contribution 1 | Context-relative, fold-safe landing-impact ranking formulation and label construction (Methodology). |
| Contribution 2 | Reproducible candidate-window scoring benchmark with matched neural and tabular baselines (Experiments). |
| Contribution 3 | Event-aware empirical findings on ranking, calibration, shortcuts, and rally difficulty, including honest negative results (Results and Discussion). |

## 4. Evaluation lock

### 4.1 Cohort

Every compared model uses exactly the same rows. A row is eligible only when:

1. its subject identifier is known;
2. its pose/statistical feature record is valid;
3. `context_high_impact_q75` is finite; and
4. `peak_fz_context_centered` is finite.

Unused impulse and loading-rate targets must not filter the cohort. Before training, a cohort manifest must record the fold, role, subject, trial, camera view, unique-impact key, targets, eligibility decision, and exclusion reason. Training must fail if compared models receive different test keys or labels.

The currently expected corrected test cohort contains 17,267 views and 2,242 unique impacts across ten held-out subjects. These counts are a preflight assertion, not a result to optimize.

### 4.2 Splits and tuning

- Outer evaluation: ten leave-one-subject-out folds.
- Inner validation: stratified by the binary label and grouped by unique-impact key; no camera view of one impact may cross train/validation, and both partitions must contain both classes.
- Label thresholds, feature normalization, early stopping, classification thresholds, and calibration are fit without the held-out subject.
- The random seed, model configuration, maximum epochs, patience, auxiliary-loss weight, and checkpoint rule are frozen in the resolved run configuration.
- A fixed seed is permitted, but fold standard deviation must be described as observed across fold-specific fits; it is not a pure estimate of between-subject variability.

### 4.3 Outcomes

- **Primary operational unit:** unique impact.
- **Primary discrimination endpoint:** fold-wise unique-impact AUROC and the paired CN-HiLDNet minus HGB difference with a 95% confidence interval.
- **Application diagnostic:** within each held-out subject and `trial_id`, aggregate camera views to one impact, rank impacts, and evaluate the top 20% review budget. Report Precision@20%, Recall@20%, and NDCG@20%; groups without positives are excluded from Recall/NDCG denominators and reported separately.
- **Secondary endpoints:** view-level AUROC/AUPRC, calibration, rally/non-rally subgroups, and auxiliary peak regression.
- **Statistical unit:** held-out-subject fold. Correlated views are not treated as independent evidence for the primary comparison.
- **Wilcoxon rule:** two-sided exact signed-rank when paired fold differences contain no zeros; otherwise use the approximate method and report the zero count. Bootstrap seed and draw count are fixed in configuration.

### 4.4 Interpretation regardless of outcome

- If CN-HiLDNet improves unique-impact results with uncertainty excluding zero, claim evidence of a model advantage under this cohort and protocol.
- If only view-level improvement is clear, claim a single-view discrimination advantage but no demonstrated event-level superiority.
- If HGB matches or exceeds CN-HiLDNet, retain the supervision/evaluation contribution and recommend HGB where simplicity or calibration matters.
- Never select a more favorable quantile, seed, review budget, subgroup, or aggregation rule after observing the corrected test results.

## 5. Self-consistency checks

- Limitations → Goal: **PASS**. The goal changes the task from absolute regression to context-relative ranking and makes subject/event handling explicit.
- Goal → Challenges: **PASS**. Context-dependent distributions, multimodal scoring, and event-aware evaluation arise directly from the goal.
- Challenges → Methodology: **PASS**. Fold-safe labels, matched scoring models, and event-aware evaluation map one-to-one.
- Methodology → Contributions: **PASS**. Each contribution is backed by one module and a named paper section.

## 6. Gates before a long run

No corrected long run starts until all gates pass:

- [x] cohort manifest and expected counts pass;
- [x] no test-subject information enters labels, transforms, thresholds, calibration, or checkpoint selection;
- [x] one config produces one self-contained run directory;
- [x] code, config, input-file hashes, environment, and seed are recorded;
- [x] a one-fold, one-epoch smoke run completes from a clean command;
- [x] smoke predictions from every compared model have identical test keys and labels;
- [x] all unit and synthetic integration tests pass;
- [x] `badminton-impact artifacts` reads only the selected run directory, not hard-coded historical paths.

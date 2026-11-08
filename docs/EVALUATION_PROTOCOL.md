# Retrospective evaluation v1

The source and `configs/evaluation_protocol.json` are committed before the evaluation run. The runner computes A/validation operating thresholds, writes and read-verifies an immutable `FROZEN_PROTOCOL.json` in private Drive, then permits the B prediction cache to open raw records only with that matching protocol. Frozen model/calibrator hashes, source fingerprint, corpus, prompts, sampling indices, software versions and scoring rules are recorded. No model fitting, feature selection, calibration fitting or threshold selection uses B. Earlier descriptive B auditing was not used to choose these rules.

## Predictive performance

Compare the already-selected LSTM with the frozen logistic baseline from the same completed training run. Preserve A/train preprocessing. Fit or restore each predeclared nonnegative sigmoid calibrator on A/calibration. Scores use FP32 frozen-model logits and float64 sigmoid calibration. No predictor retraining occurs.

Report hourly AUROC, non-interpolated average precision, prevalence, Brier score, log loss and ten-bin equal-width ECE for raw and calibrated scores. Calibrated operating points are: fixed 0.5; the largest A/validation score cutoff attaining at least 80% hourly recall (ties retained); and the maximum-utility A/validation cutoff among 51 score quantiles plus all/none decisions, preferring the higher cutoff on ties. Raw 0.5 is retained as a historical weighted-training reference. Operating points are not chosen again on B.

Resample whole patient-record files, with replacement, 500 times using seed 42. Preserve every hour within a sampled file. Compute percentile 95% confidence intervals for calibrated AUROC, AP, Brier, log loss, sensitivity/PPV at the validation recall rule, and utility at the validation utility rule. Use the same draws for the paired LSTM-minus-logistic AP interval. These intervals condition on the frozen models/calibrators/thresholds; they do not capture training uncertainty or undocumented linkage between patient IDs.

Report hourly confusion counts, sensitivity, specificity, PPV and F1; patients with any alarm; and false hourly flags and alarm episodes per 100 nonseptic patient-hours. Alarm episodes use a six-hour refractory interval. Patient-ever-sepsis sensitivity allows an alarm at any time and must not be described as early sensitivity. Separately report detection at/before the first positive shifted label and label-relative lead time among detected septic records; records positive at their first hour are counted as left-censored. These label-clock summaries are not directly measured clinical-onset lead times.

Challenge utility uses the official PhysioNet 2019 convention and parameters (-12, -6, +3 hours; TP maximum 1, FN minimum -2, FP -0.05, TN 0), with normalization relative to inaction and the best possible decisions. Labels are already six-hour shifted and are never shifted again. The utility implementation is checked against the pinned upstream function, commit `467c49b514542be7a4a0bafe40fa2c3b064dda2e`, under its BSD-2-Clause license. Results on this public B split are not official hidden-test leaderboard scores.

## Patient-specific explanation quality

Preserve the existing Qwen3.8-27B revision, prompts, corpus, BF16 decoding, six-call/two-retrieval limits and a 20-minute inference budget. Choose one label-independent hashed-time endpoint per A/validation patient, then select 12 unique patients: four calibrated-risk quartiles, each with low, median and high observed-window missingness. Selection occurs before B evaluation and uses no outcome labels. This is a stratified development sample, not a representative population estimate.

Audit the two saved three-case runs and the new sample. Report output/abstention coverage, exact citation/numeric-check validity, use of actual observed feature values with explicit patient-context language, feature-grounding coverage, query diversity, duplicate outputs and pairwise text similarity. Numeric grounding uses a predeclared 0.05 absolute or 0.5% rounding tolerance. Both all-case and eligible-observation denominators are retained. A valid citation alone does not establish patient-specific usefulness or independent entailment.

Compute input-channel ablation sensitivity for the selected predictor: remove each feature's value and observation-mask channel across the causal window by setting both to zero. Compare grounded feature mentions with the five largest absolute probability changes and report how many of those features the generator could actually see. This is an input-sensitivity proxy, not a causal physiological intervention or a ground-truth clinical explanation.

Prepare a private, randomly ordered review packet containing causal observations and explanations/cited passages. Exclude original IDs, outcome labels and model identity; keep the private mapping separate. Blank clinician ratings cover relevance, specificity, observation accuracy, evidence support and misleading/unsafe content. No LLM or software fills clinical ratings. Independent clinician assessment and agreement analysis remain pending until real annotations are supplied.

## Interpretation and persistence

B is a retrospective held-out source from the public PhysioNet 2019 development data. It is not a prospective deployment study. Once scored, it is consumed for this fixed comparison and cannot be treated as a fresh test set for later model selection. Agent explanations do not alter the frozen predictor's scores. Clinical effectiveness, treatment decisions, prospective alarm benefit and independent expert-rated explanation quality are not established by this evaluation alone.

Private Drive retains frozen protocol, predictions, bootstrap draws, case traces, sensitivity outputs and reviewer material under `evaluations/<evaluation-id>`. Public source control receives methods, aggregate metrics and aggregate figures only. No raw or per-patient derivatives are committed.

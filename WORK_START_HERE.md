# SepsisGuard-UAR — Working handoff

Use one Run-all workflow, without calendar-based handoffs: `notebooks/SepsisGuard_UAR.ipynb`. The current notebook executes `scripts.run_evaluation`; preparation and predictive training are complete. Colab Pro is the main compute environment, private Drive holds protected outputs, and https://github.com/shakiful/Sepsisguard-Uar is canonical source.

## Current checkpoint

- Preparation: `pipeline-20261104T190221Z-22906fa2`, observed A100 80GB / 167.05 GiB RAM. Cohort 40,336 patients / 1,552,210 hours; 39 retained predictors; causal 12 x 78 windows. Exact original partition preserved.
- Training: `training-05ed8fa8a5f01f4f92a8`, observed A100 40GB / 83.47 GiB RAM, CUDA BF16. Selected LSTM best epoch 1, validation AP 0.098821 / AUROC 0.772333. Preserve `lstm_best.pt` SHA256 `266d203773981dce01f2b604c29895e4588b0a2568aa16a333d6a6ffa62764bc`.
- Agent implementation: pinned official Qwen3.8-27B, BF16, one shared native model; LangGraph retrieval, drafting, verification and abstention. Reserved A/calibration supplies sigmoid calibration. The frozen training result remains uncalibrated.
- Real agent run: `agents-dc179bf37a3ea479c35b`, source `cd6c601de5ec7f028ba6a654706159114a76b2cd`, verified directly in connected Colab on 2026-11-08. Native Qwen3.8-27B BF16; A100 80GB / 167.05 GiB RAM; model-load peak allocation 50.96 GiB. Three cases completed planner/retrieval/writer/verifier with nine model calls and six accepted claims. The first planner-only run `agents-b5667526ece3287af39f` abstained three times and is preserved. See `reports/verification/AGENT_COLAB_EVIDENCE.json`.
- Calibration: fitted on 119,051 A/calibration hours (2,574 positive), non-flat sigmoid. Fit Brier 0.020253 / log loss 0.090618 are fit diagnostics only. The predictor was not retrained; B was not used.
- Fixed corpus: 19 bundled CDC/NIH passages, source hashes match the earlier live fetch. Default retrieval needs no website requests; intentional live refresh must be recorded as a new corpus version. Both new source commits passed all 56 tests in CI.

## Continue directly

1. Open the current GitHub notebook in Colab; old Drive copies retain old code. Choose the user-authorized A100 **80GB** / about **160GB RAM** and Run all. Preflight stops before weights on a smaller allocation. Record actual hardware rather than assuming the requested allocation.
2. Mount the same private Drive containing the completed runs above. The runner verifies both frozen predictors and the preparation snapshot, calibrates only on A/calibration, selects thresholds only on A/validation, and saves/read-verifies `FROZEN_PROTOCOL.json` before B predictive evaluation. No predictor retraining is needed.
3. First load downloads about 54–56GB to runtime disk. A 20-minute inference budget follows loading. One shared model uses an 8K context cap and bounded sequential calls. Reference snapshots, calibration and case checkpoints persist privately.
4. Review `MyDrive/SepsisGuard-UAR/evaluations/<evaluation-id>/EVALUATION_RESULT.json` and `EVALUATION_REPORT.md`. The runner scores B with fixed rules and audits 12 unique A/validation patients. Private predictions, input-ablation sensitivity and randomized blank clinician review material persist there. Independent clinical ratings are pending; same-model verification does not prove correctness.
5. Connected browser access and private Drive mount now work. The user explicitly approved the Google Drive for desktop OAuth connection used by this Colab mount. The earlier live reference fetch returned HTTP 403; bundled snapshots fixed this without changing reference text. Local CPU-only checks remain separate from actual Colab GPU results. The configuration profile records the initial declared experiment settings; execution observations belong in the evidence file.
6. The three outputs were general sepsis context and all queries/passages were identical. Do not treat them as patient-specific explanations or evidence of uncertainty-routing benefit. The fixed evaluation now audits varied causal cases and prepares blinded annotation, preserving both initial runs. Budget-matched agent comparisons remain future work.

## Research invariants

A/train determines preprocessing and predictor fitting; A/validation supports selection and development; A/calibration fits only the predeclared calibration method; B remains locked until final protocol freeze. Original PhysioNet labels are already shifted and must not be shifted again. Filling/windows reset at patient boundaries. Prompts exclude labels, future observations and patient identifiers. B descriptive EDA must not guide features, prompts, thresholds or model selection.

Public GitHub holds source, tests and aggregate evidence. Never commit patient records, manifests, per-file hashes, fitted artifacts, predictions, case traces or weights. The agent uses local CDC/NIH snapshots, no hosted inference or external tracing, and returns cited research context or abstention. No clinical-use claim is established.

Next evaluation: compare budget-matched RAG, uncertainty routing and verification/abstention with fixed predictor/corpus/decoding. Blinded annotations are needed for explanation correctness; same-backbone judgments are not independent truth. A tiny functional run does not establish improved AP, clinical benefit or publication readiness.

## Fixed evaluation protocol

See `docs/EVALUATION_PROTOCOL.md` and `configs/evaluation_protocol.json`. Evaluation implementation passed 64 tests locally. Source/configuration are committed before the real run; do not claim measured evaluation results until `EVALUATION_RESULT.json` completes and private readback is verified. Once B is predictively scored it is consumed for this comparison; no B-driven retuning is permitted.

# SepsisGuard-UAR — Working handoff

Use one Run-all workflow, without calendar-based handoffs: `notebooks/SepsisGuard_UAR.ipynb`. The current notebook executes `scripts.run_agents`; preparation and predictive training are complete. Colab Pro is the main compute environment, private Drive holds protected outputs, and https://github.com/shakiful/Sepsisguard-Uar is canonical source.

## Current checkpoint

- Preparation: `pipeline-20261104T190221Z-22906fa2`, observed A100 80GB / 167.05 GiB RAM. Cohort 40,336 patients / 1,552,210 hours; 39 retained predictors; causal 12 x 78 windows. Exact original partition preserved.
- Training: `training-05ed8fa8a5f01f4f92a8`, observed A100 40GB / 83.47 GiB RAM, CUDA BF16. Selected LSTM best epoch 1, validation AP 0.098821 / AUROC 0.772333. Preserve `lstm_best.pt` SHA256 `266d203773981dce01f2b604c29895e4588b0a2568aa16a333d6a6ffa62764bc`.
- Agent implementation: pinned official Qwen3.8-27B, BF16, one shared native model; LangGraph retrieval, drafting, verification and abstention. Reserved A/calibration supplies sigmoid calibration. The frozen training result remains uncalibrated.
- Actual agent validation: scripted synthetic end-to-end execution and integrity/resume tests; live CDC/NIH retrieval and pinned tokenizer/chat-template check. **No real Qwen weight load or generation yet.** See `reports/verification/AGENT_IMPLEMENTATION_VALIDATION.json`.

## Continue directly

1. Open the current GitHub notebook in Colab; old Drive copies retain old code. Choose the user-authorized A100 **80GB** / about **160GB RAM** and Run all. Preflight stops before weights on a smaller allocation. Record actual hardware rather than assuming the requested allocation.
2. Mount the same private Drive containing the completed runs above. The runner verifies frozen weights and prepared snapshot, restores/downloads raw data, fits calibration only on A/calibration, and runs three deterministic A/validation endpoints. No predictor retraining is needed.
3. First load downloads about 54–56GB to runtime disk. A 20-minute inference budget follows loading. One shared model uses an 8K context cap and bounded sequential calls. Reference snapshots, calibration and case checkpoints persist privately.
4. Review `MyDrive/SepsisGuard-UAR/agents/<run-id>/AGENT_RESULT.json`, `AGENT_REPORT.md`, `backend_calls.json` and case traces. Record supported-context and abstention counts, failures, citations, latency and measured GPU allocation. Workflow completion and the same-model verifier do not prove explanation correctness.
5. Browser attempts fail `Cannot attach to this target`; local real-model preflight fails because this host has no CUDA. If still blocked, report that and inspect user-supplied execution output. Never claim real inference from scripted tests or tokenizer checks.

## Research invariants

A/train determines preprocessing and predictor fitting; A/validation supports selection and development; A/calibration fits only the predeclared calibration method; B remains locked until final protocol freeze. Original PhysioNet labels are already shifted and must not be shifted again. Filling/windows reset at patient boundaries. Prompts exclude labels, future observations and patient identifiers. B descriptive EDA must not guide features, prompts, thresholds or model selection.

Public GitHub holds source, tests and aggregate evidence. Never commit patient records, manifests, per-file hashes, fitted artifacts, predictions, case traces or weights. The agent uses local CDC/NIH snapshots, no hosted inference or external tracing, and returns cited research context or abstention. No clinical-use claim is established.

Next evaluation: compare budget-matched RAG, uncertainty routing and verification/abstention with fixed predictor/corpus/decoding. Blinded annotations are needed for explanation correctness; same-backbone judgments are not independent truth. A tiny functional run does not establish improved AP, clinical benefit or publication readiness.

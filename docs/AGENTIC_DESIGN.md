# SepsisGuard-UAR agent layer — proposed design

Status: design and model-selection profile only. No agent runtime, LLM download, LLM fine-tuning or agent performance result is claimed here. Keep the existing single Colab notebook as the eventual entrypoint.

## Frozen predictive baseline

Preserve training run `training-05ed8fa8a5f01f4f92a8` and selected `lstm_best.pt` SHA256 `266d203773981dce01f2b604c29895e4588b0a2568aa16a333d6a6ffa62764bc`. The selected LSTM's validation AP is 0.098821 and AUROC 0.772333. Its scores are not calibrated. B and calibration raw files were not opened by training. The explanation layer must not overwrite these scores or represent the validation metrics as independent test performance.

## Local model choice

As of the 2026-11-04 source check, the official `Qwen/Qwen3.8-27B` checkpoint is a verified newer 27B-family candidate than Qwen3.5-27B, with official agentic-task support. It is the proposed primary backbone. Keep `Qwen/Qwen3.5-27B` as a same-size comparison and evaluate Gemma 4 31B as an alternative. General model-card benchmarks do not establish which is best for this sepsis research task.

The user reports an available configuration of 80GB VRAM and about 160GB system RAM. This is a requested future allocation, distinct from the observed training allocation of A100 40GB / 83.47 GiB RAM. Start with BF16 inference, batch size 1 and an 8,192-token context cap. A nominal 27B dense model requires about 54GB for two-byte weights alone; the Qwen Hub page reports about 28B total parameters, so allow roughly 54–56GB before KV cache, runtime workspaces and other allocation overhead. Measure actual load/generation peaks before increasing context. A 40GB allocation would require a separately tested quantized/offloaded configuration rather than assuming BF16 fits.

Use one loaded Qwen instance with distinct role prompts and sequential calls. Loading separate 27B copies for planner, retriever and verifier is unnecessary. Prefer text-only processing for tabular summaries and reference passages. Pin the official model revision and tested library versions after a loader/tool-call smoke test. Prefer safetensors and supported built-in model classes; do not silently enable remote executable code.

Keep downloaded base weights on the Colab runtime disk initially; free Drive capacity for a roughly 54–56GB source checkpoint has not been checked. Persist model ID/revision, prompts, retrieval corpus versions, traces, evaluation results and any later adapters privately in Drive. No external hosted inference, tracing or embedding endpoint is configured by this design.

## Two uses of agents

1. A bounded experiment assistant can propose configurations, call the existing trainer and summarize validation histories. It may select only from explicitly bounded experiments, with a declared GPU/time budget. It cannot change patient splits, use calibration/B for model selection, execute arbitrary generated shell code or silently launch unlimited trials. This is optional research automation, not evidence of improved predictive accuracy.
2. The main SepsisGuard-UAR agent layer uses the frozen predictor as a tool and performs uncertainty-conditioned retrieval, evidence checking and selective abstention. An LLM's self-reported confidence is not a substitute for calibrating the numerical predictor.

## Proposed workflow

- **Data/predictor tools:** validate feature/timestamp availability, apply frozen causal preprocessing, and return the predictor's score with immutable model/input provenance. Include only information available at that endpoint. No labels, future observations or patient identifiers enter prompts.
- **Calibration/uncertainty gate:** fit a predeclared calibration method using reserved A/calibration, then evaluate the intended uncertainty estimator and routing policy on development data. Until this exists, report raw scores as uncalibrated and avoid a fabricated confidence guarantee.
- **Retrieval agent:** choose search queries and, when useful, one reformulation against a versioned local corpus of public, appropriately licensed references. Return exact source and passage identifiers; two retrieval attempts maximum.
- **Draft and verifier:** produce structured claims with passage citations. Check source existence, quoted spans, numerical consistency with predictor tools and whether each claim is actually supported. A second role prompt using the same model is not independent evidence of correctness; add programmatic checks and blinded annotated evaluation.
- **Output/abstention:** publish a cited research explanation only when checks pass; otherwise return a structured insufficient-evidence/uncertainty status. Six tool decisions, two retrieval attempts, 512 final-answer tokens and 1,024 total generated tokens per call are provisional initial caps, to be benchmarked before use.

LangGraph is the proposed state/routing/checkpoint runtime. The dynamic parts are bounded query/tool choices and evidence-driven retries, rather than simply renaming a fixed preprocessing chain as agents. Start with the official tool-calling/chat template and structured output smoke tests for the selected Qwen revision. Use its supported non-thinking mode initially, with the generation cap covering all generated tokens; evaluate a bounded reasoning mode separately.

## Evaluation

Compare the predictor alone, budget-matched standard RAG, uncertainty-conditioned RAG, and retrieval plus verification/abstention. Hold the predictor, reference corpus, decoding settings and comparable token/retrieval budgets fixed where the comparison requires it. If comparing LLM backbones, log their revisions, dtype/quantization, prompts and serving engines.

Measure unsupported-claim rate, citation validity and entailment, tool-call/schema success, abstention coverage, error at retained coverage, latency and GPU memory. Do not use the same model's self-judgment as the sole ground truth. Prediction AP will remain the frozen predictor's AP unless a different prediction method is separately implemented and evaluated. Keep B locked until models, calibration, prompts, retrieval corpus, budgets and evaluation protocol are frozen.

## Primary sources checked

- https://huggingface.co/Qwen/Qwen3.8-27B
- https://github.com/QwenLM/Qwen3.8
- https://huggingface.co/Qwen/Qwen3.5-27B
- https://ai.google.dev/gemma/docs/core/model_card_4
- https://huggingface.co/docs/transformers/quantization/bitsandbytes
- https://docs.langchain.com/oss/python/langgraph/workflows-agents
- https://docs.langchain.com/oss/python/langgraph/overview

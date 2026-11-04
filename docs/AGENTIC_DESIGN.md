# SepsisGuard-UAR evidence agent

Status: implemented and tested; real Qwen GPU execution pending. The single Colab notebook is the entrypoint. No LLM fine-tuning is performed.

## Frozen inputs and calibration

`scripts/run_agents.py` restores training `training-05ed8fa8a5f01f4f92a8` and preparation `pipeline-20261104T190221Z-22906fa2`. Selected LSTM weights must match SHA256 `266d203773981dce01f2b604c29895e4588b0a2568aa16a333d6a6ffa62764bc`. Manifest, model signature, preprocessing provenance and consumed raw bytes are checked. Previously measured validation AP is 0.098821 and AUROC 0.772333; the original predictor is unchanged.

The runner explicitly permits reserved A/calibration, collects frozen FP32 logits and fits unweighted sigmoid calibration with nonnegative slope using bounded convex optimization. A zero slope is reported as a flat calibrator rather than reversing ordering. B is never used for model inference, calibration or agent prompts. Fit Brier/log loss are fit diagnostics, not independent evaluation. Initial routing thresholds (entropy 0.3 bits or missingness 0.5) are fixed development heuristics, not validated operating points.

Three deterministic spread-out A/validation endpoints provide the initial run without label-based selection. Records are sliced before transformation and summary. Prompts receive selected current measurements, a case index, predictor scores, entropy and missingness. They exclude original IDs, labels and future rows. Download/restoration may extract B archives; B records are not consumed for calibration or agent inference.

## Model and execution bounds

Official `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` is pinned. Transformers 5.22.1 supplies `Qwen3_5ForConditionalGeneration`; `AutoTokenizer` avoids video processors for text inputs. Native class import and pinned tokenizer/chat template passed locally. Actual weights, GPU operators, peak memory and generated output remain unmeasured.

The user reports 80GB GPU / about 160GB RAM. Preflight requires CUDA, 70 GiB total and 64 GiB free. BF16 weights need roughly 54–56GB before other allocations. No automatic quantization, CPU offload or smaller-GPU fallback exists. Base weights use runtime `model-cache/`; revision and provenance persist privately. One shared native safetensors model serves all three roles sequentially; remote executable code is disabled.

Inputs plus output must fit 8,192 tokens. Greedy non-thinking caps are 128 planner / 512 writer / 256 verifier tokens, at most 120 seconds per call. The initial inference budget is 20 minutes after loading. No hosted inference, external embedding or tracing is configured. Failed parsing, timeout, unsupported content or insufficient evidence leads to abstention.

## Graph and evidence checks

LangGraph runs plan → local retrieval → draft → verify. The planner can search or abstain; it cannot execute shell, network calls or unrestricted tools. Each case permits six model calls and two retrieval attempts at most. Lower-entropy, lower-missingness cases allow one attempt. Failed citation/entailment checks can trigger one query reformulation within budget.

Local TF-IDF retrieval uses versioned CDC and NIH NIGMS educational text. Source hashes, passage IDs and timestamps persist; cached passages must match snapshot bytes. This is a small reference corpus, not a comprehensive guideline collection. Evidence text is untrusted data and cannot grant tool permissions.

Writers produce at most two structured claims. Programmatic checks require known IDs, exact 3–25-word quoted spans, at most 25 quoted words per source, consistent numbers and bounded schemas; common diagnostic/treatment assertions are rejected. The verifier checks entailment for every claim using the same backbone. It is not independent factual certification, and filters do not prove every unsafe or incorrect paraphrase is caught.

Passing outputs are `supported_research_context`; other paths are `abstained`. Numerical risk comes from predictor/calibration code, not generated claims. This feature establishes no diagnosis or treatment recommendation.

## Persistence and validation

Run identities include frozen weights, calibration, source fingerprint, indices, corpus, profile and model/library revisions. Case files are copied and hash-read back before immutable pointers are published. Matching completed cases resume; changed cached bytes are rejected. Private result/report files, role traces, calibrator and snapshots support review.

The CPU demo trains a tiny synthetic predictor and uses scripted role responses. It exercises calibration, graph transitions, persistence, readback and resume; it does not load or generate with Qwen. Live CDC/NIH retrieval and pinned tokenizer checks were separate. See `reports/verification/AGENT_IMPLEMENTATION_VALIDATION.json` for aggregate evidence and blockers.

## Evaluation still required

Compare predictor alone, budget-matched standard RAG, uncertainty routing and verification/abstention with fixed predictor, corpus and decoding. Measure unsupported claims, citation/entailment correctness, schema success, coverage, selective error, latency and peak memory. Use blinded annotations rather than same-model judgments as sole ground truth. Freeze prompts, corpus, calibration and protocol before B evaluation. Implementation alone does not improve or remeasure AP. A bounded experiment-automation assistant and alternative LLM comparisons are not implemented.

## Primary references

- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/docs/transformers/model_doc/qwen3_5
- https://docs.langchain.com/oss/python/langgraph/overview
- https://www.cdc.gov/sepsis/about/index.html
- https://www.nigms.nih.gov/education/fact-sheets/Pages/sepsis.aspx

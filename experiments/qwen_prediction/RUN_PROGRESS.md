# Qwen experiment completed and independently verified

The earlier connection blocker was resolved. The real run `qwen-lora-a98c78da4169245666a6` completed both epochs (256 updates), frozen A-only calibration/thresholds, and all 1,024 matched B predictions for both base and fine-tuned Qwen.

**No clear fine-tuning AP benefit was demonstrated.** Calibrated pilot AP: base Qwen 0.03265, LoRA Qwen 0.03192, saved LSTM 0.06246, saved logistic 0.11141. These are exploratory, weighted sampled estimates, not replacement full-cohort results.

Read [the completed interpretation and limitations](reports/qwen-lora-a98c78da4169245666a6/RESULTS.md), [exact aggregate result](reports/qwen-lora-a98c78da4169245666a6/QWEN_RESULT.json), and [independent readback](reports/qwen-lora-a98c78da4169245666a6/POST_RUN_READBACK.json).

The private run directory remains `MyDrive/SepsisGuard-UAR/experiments/qwen_prediction/qwen-lora-a98c78da4169245666a6`. Inputs, selected adapter, optimizer state, calibration and individual scores are preserved there. Original `main`, notebooks and results remain unchanged.

Historical progress is retained in commit `be2fce6cc525ca2cb4cbc2d1169e1d3f17ad294d`: the final observation before the first browser loss was update 252. Later direct Drive inspection proved that training had actually finished and 680 LoRA B scores had persisted. Colab's temporary provisioning failure supplied an insufficient L4; no model computation was launched on it. Exact source/runtime identity was restored on A100 80GB, and the remaining 344 scores completed without repeating optimization or changing the frozen protocol.

Executed source: `5bb4b1dc52d726596af8144357e4c9235e0e64e9`. Frozen evaluation hash: `77711c5a94f8008f9439dca77615f9746b273baa3733e7b33308179fc47204b7`. All four matched metric sets, original result/prediction hashes, completed checkpoint, and all sampled past-only inputs passed the independent audit.

[Executed separate Colab notebook](https://colab.research.google.com/drive/1KTxQ3rddV9CA_G--sZ4PnkxxiO0eM7q8).

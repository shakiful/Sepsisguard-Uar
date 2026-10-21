"""CPU preprocessing fit and development-split smoke checks; B stays unopened."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sepsisguard.dataset import CausalWindowDataset, seed_worker, seeded_generator
from sepsisguard.preprocessing import TrainOnlyPreprocessor


def run(data_root, manifest_path, output, min_observed_fraction=0.0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    pre = TrainOnlyPreprocessor.fit(data_root, manifest,
            min_observed_fraction=min_observed_fraction, scratch_dir=output)
    pre.save(output / "preprocessor.json")
    metrics = {}
    for split in ("train", "validation", "calibration"):
        dataset = CausalWindowDataset(data_root, manifest, pre, split)
        # Deterministic spread across patients, including first and last hours.
        indices = np.unique(np.linspace(0, len(dataset) - 1, min(128, len(dataset)), dtype=int))
        for index in indices:
            item = dataset[int(index)]
            assert torch.isfinite(item["x"]).all()
            assert int(item["time_mask"].sum()) == item["length"]
            assert item["y"].item() in (0, 1)
        loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0,
                            generator=seeded_generator(42), worker_init_fn=seed_worker)
        batch = next(iter(loader))
        metrics[split] = {"patients": len(dataset.rows), "hourly_windows": len(dataset),
                          "windows_checked": len(indices), "batch_shape": list(batch["x"].shape)}
    result = {"status": "preprocessing_and_dataset_checks_passed", "window_hours": 12,
              "selected_features": pre.features, "input_channels": len(pre.features) * 2,
              "fit": pre.provenance, "splits": metrics, "test_B_opened": False,
              "torch_version": torch.__version__,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
              if (Path.cwd() / ".git").exists() else "source ZIP; see starter hash in run metadata"}
    (output / "WEEK2_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--manifest", default="reports/week1/split_manifest.csv")
    p.add_argument("--output", default="reports/week2")
    p.add_argument("--min-observed-fraction", type=float, default=0.0)
    args = p.parse_args()
    run(args.data_root, args.manifest, args.output, args.min_observed_fraction)

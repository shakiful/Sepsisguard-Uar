"""Locked patient-level development/calibration/holdout partitions."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED = {"patient_id", "source", "n_hours", "ever_sepsis"}


def check_split_integrity(manifest: pd.DataFrame) -> None:
    required = REQUIRED | {"split"}
    if missing := required.difference(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if manifest["patient_id"].isna().any() or manifest["patient_id"].duplicated().any():
        raise ValueError("Patient ID overlap or missing patient IDs")
    if not manifest["source"].isin(["A", "B"]).all():
        raise ValueError("Unknown or missing source")
    lengths = pd.to_numeric(manifest["n_hours"], errors="coerce")
    if lengths.isna().any() or not ((lengths > 0) & (lengths % 1 == 0)).all():
        raise ValueError("Invalid patient record lengths")
    if not manifest["ever_sepsis"].isin([0, 1]).all():
        raise ValueError("Invalid patient outcome")
    if not manifest["split"].isin(["train", "validation", "calibration", "test_B"]).all():
        raise ValueError("Unexpected split name")
    if (manifest.query("source == 'B'")["split"] != "test_B").any():
        raise ValueError("B patients must be reserved for test_B")
    if (manifest.query("source == 'A'")["split"] == "test_B").any():
        raise ValueError("A patients cannot be in test_B")
    if not manifest["patient_id"].astype(str).str.startswith(("A:", "B:")).all():
        raise ValueError("Expected dataset-qualified patient IDs")
    if any(not patient_id.startswith(f"{src}:") for patient_id, src in zip(manifest["patient_id"], manifest["source"])):
        raise ValueError("Patient ID's source prefix disagrees with source column")


def make_splits(summary: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    if missing := REQUIRED.difference(summary.columns):
        raise ValueError(f"Summary missing columns: {sorted(missing)}")
    if summary["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs in EDA summary")
    if not summary["source"].isin(["A", "B"]).all():
        raise ValueError("Unknown or missing source")
    a = summary.loc[summary["source"].eq("A")].copy()
    b = summary.loc[summary["source"].eq("B")].copy()
    if a.empty or b.empty:
        raise ValueError("Expected both A (development) and B (held-out site) datasets")
    counts = a["ever_sepsis"].value_counts()
    if set(counts.index) != {0, 1} or counts.min() < 10:
        raise ValueError("Dataset A needs at least 10 patients of each class for stratified 70/15/15 splitting. Use more patients, not just a tiny smoke-test subset.")
    a_train, a_other = train_test_split(
        a, test_size=0.30, random_state=seed, stratify=a["ever_sepsis"])
    a_val, a_cal = train_test_split(
        a_other, test_size=0.50, random_state=seed, stratify=a_other["ever_sepsis"])
    chunks = []
    for frame, name in ((a_train, "train"), (a_val, "validation"),
                        (a_cal, "calibration"), (b, "test_B")):
        t = frame.copy()
        t["split"] = name
        chunks.append(t)
    manifest = pd.concat(chunks, ignore_index=True).sort_values("patient_id").reset_index(drop=True)
    check_split_integrity(manifest)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Create reproducible patient-level A train/val/cal + B holdout")
    p.add_argument("--summary", default="reports/week1/patient_summary.csv")
    p.add_argument("--output", default="reports/week1/split_manifest.csv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    manifest = make_splits(pd.read_csv(args.summary), args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    print("Wrote", path)
    print(manifest.groupby(["source", "split"]).agg(patients=("patient_id", "size"), positive_patients=("ever_sepsis", "sum")))

if __name__ == "__main__":
    main()

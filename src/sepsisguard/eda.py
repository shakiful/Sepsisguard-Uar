"""Streaming descriptive EDA; patient summaries and aggregate plots only."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .load import iter_patients
from .schema import FEATURES, LABEL, EXPECTED_PATIENT_COUNTS


def run_eda(data_root: str | Path, output_dir: str | Path, max_per_source: int | None = None) -> dict:
    out = Path(output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    missing_counts = pd.Series(0, index=FEATURES, dtype="int64")
    seen_count = pd.Series(0, index=FEATURES, dtype="int64")
    summaries = []
    total_rows = 0
    positive_hours = 0
    for df in iter_patients(data_root, max_per_source=max_per_source):
        n = len(df)
        total_rows += n
        positive_hours += int(df[LABEL].sum())
        missing_counts += df[FEATURES].isna().sum().astype("int64")
        seen_count += n
        positive_idx = np.flatnonzero(df[LABEL].to_numpy() == 1)
        summaries.append({
            "patient_id": df["patient_id"].iat[0],
            "source": df["source"].iat[0],
            "n_hours": n,
            "n_positive_hours": int(df[LABEL].sum()),
            "ever_sepsis": int(len(positive_idx) > 0),
            "first_positive_hour_index": int(positive_idx[0]) if len(positive_idx) else np.nan,
        })
    if not summaries:
        raise ValueError("No patients were loaded")
    patients = pd.DataFrame(summaries).sort_values("patient_id").reset_index(drop=True)
    patients.to_csv(out / "patient_summary.csv", index=False)
    missing = pd.DataFrame({
        "feature": FEATURES,
        "missing_count": missing_counts.to_numpy(),
        "total_hours": seen_count.to_numpy(),
        "missing_fraction": (missing_counts / seen_count).to_numpy(),
    }).sort_values("missing_fraction", ascending=False).reset_index(drop=True)
    missing.to_csv(out / "feature_missingness.csv", index=False)
    class_table = patients.groupby("source", as_index=False).agg(
        patients=("patient_id", "size"),
        sepsis_patients=("ever_sepsis", "sum"),
        hours=("n_hours", "sum"),
        positive_hours=("n_positive_hours", "sum"),
    )
    class_table["sepsis_patient_fraction"] = class_table["sepsis_patients"] / class_table["patients"]
    class_table["positive_hour_fraction"] = class_table["positive_hours"] / class_table["hours"]
    class_table.to_csv(out / "dataset_summary.csv", index=False)
    lengths = patients.groupby("source")["n_hours"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    lengths.to_csv(out / "record_length_summary.csv")

    fig, ax = plt.subplots(figsize=(11, 5))
    m = missing.sort_values("missing_fraction", ascending=True)
    ax.barh(m["feature"], m["missing_fraction"])
    ax.set(xlabel="Missing fraction of hourly observations", title="PhysioNet 2019 feature missingness", xlim=(0, 1))
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "figures" / "feature_missingness.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    class_table.plot.bar(x="source", y=["patients", "sepsis_patients"], ax=ax, rot=0)
    ax.set(ylabel="Patients", title="Patient-level sepsis prevalence")
    fig.tight_layout()
    fig.savefig(out / "figures" / "patient_prevalence.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(patients["n_hours"], bins=25)
    ax.set(xlabel="Hours recorded per patient", ylabel="Patient count", title="ICU record lengths")
    fig.tight_layout()
    fig.savefig(out / "figures" / "record_lengths.png", dpi=160)
    plt.close(fig)

    source_counts = patients.groupby("source")["patient_id"].count().to_dict()
    truncated = max_per_source is not None
    completeness = all(source_counts.get(s) == expected for s, expected in EXPECTED_PATIENT_COUNTS.items())
    synthetic = (Path(data_root) / "SYNTHETIC_DATA.txt").exists()
    banner = "SYNTHETIC SMOKE TEST ONLY — no scientific results." if synthetic else "Real PhysioNet records; cohort completeness is reported below."
    report = f"""# Week 1 Exploratory Data Analysis

**{banner}**

Dataset: PhysioNet/CinC Challenge 2019 v1.0.0. Data are retrospective and the target is the **already shifted** challenge label.

- Patients examined: **{len(patients):,}**
- Hourly observations: **{total_rows:,}**
- Positive hourly observations: **{positive_hours:,}** ({positive_hours / total_rows:.2%})
- Patients with any positive hour: **{int(patients['ever_sepsis'].sum()):,}** ({patients['ever_sepsis'].mean():.2%})
- Source A: **{source_counts.get('A', 0):,}** / 20,336 expected
- Source B: **{source_counts.get('B', 0):,}** / 20,000 expected
- Full expected training cohort represented: **{'yes' if completeness else 'no'}**
- Limited smoke-test mode: **{'yes' if truncated else 'no'}**

## Five features with highest missing fractions

{missing.head(5).to_markdown(index=False) if False else chr(10).join(f'- {r.feature}: {r.missing_fraction:.1%}' for r in missing.head(5).itertuples())}

## Outputs

- `patient_summary.csv`: one row per patient, including labels and source.
- `dataset_summary.csv`: aggregated patient/hour class distributions.
- `feature_missingness.csv`: missing observation fractions per predictor.
- `figures/*.png`: aggregate visualizations.

## Methodological notes

- `SepsisLabel` was shifted in the original challenge. Never shift it again when using original evaluation.
- Patient-level prevalence and hourly class balance are **different** statistics.
- No train/validation/test preprocessing was fitted during this descriptive analysis.
- EDA on the complete source B is **descriptive only**; avoid using it to choose models, features, or thresholds when preserving B as a test set.
- This report does **not** claim predictive accuracy or clinical validation.
"""
    (out / "EDA_REPORT.md").write_text(report, encoding="utf8")
    return {"patients": len(patients), "hours": total_rows, "positive_hours": positive_hours,
            "patient_sepsis": int(patients["ever_sepsis"].sum()), "complete": completeness,
            "output": str(out)}


def main() -> None:
    p = argparse.ArgumentParser(description="Week 1 streaming PhysioNet EDA")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--output-dir", default="reports/week1")
    p.add_argument("--max-per-source", type=int, default=None,
                   help="Smoke test only; not for publishable results")
    args = p.parse_args()
    print(run_eda(args.data_root, args.output_dir, args.max_per_source))

if __name__ == "__main__":
    main()

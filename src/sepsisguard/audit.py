"""Real-cohort validation, duplicate detection, and private data fingerprints."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd

from .load import patient_files, read_patient
from .preprocessing import validate_hourly
from .schema import EXPECTED_PATIENT_COUNTS
from .splits import check_split_integrity


def audit_data(data_root, output_dir, manifest=None, expected_counts=None):
    expected_counts = EXPECTED_PATIENT_COUNTS if expected_counts is None else expected_counts
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inventory, seen, counts, digest = [], {}, Counter(), hashlib.sha256()
    mapping = None
    if manifest is not None:
        check_split_integrity(manifest)
        mapping = manifest.set_index("patient_id")
    for source, path in patient_files(data_root):
        patient_id = f"{source}:{path.stem}"
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha in seen:
            raise ValueError(f"Duplicate patient file contents: {patient_id} and {seen[sha]}")
        seen[sha] = patient_id
        patient = read_patient(path, source)
        validate_hourly(patient)
        if mapping is not None:
            if patient_id not in mapping.index:
                raise ValueError("Data patient absent from split manifest")
            row = mapping.loc[patient_id]
            if row.source != source or row.n_hours != len(patient) or row.ever_sepsis != int(patient.SepsisLabel.max()):
                raise ValueError("Manifest does not match data")
        counts[source] += 1
        digest.update(patient_id.encode() + b"\0" + bytes.fromhex(sha))
        inventory.append({"patient_id": patient_id, "sha256": sha, "bytes": len(raw), "n_hours": len(patient)})
        if len(inventory) % 5000 == 0:
            print(f"Validated {len(inventory):,} patient files", flush=True)
    if dict(counts) != expected_counts:
        raise ValueError(f"Incomplete cohort: observed {dict(counts)}, expected {expected_counts}")
    if mapping is not None and set(mapping.index) != {r["patient_id"] for r in inventory}:
        raise ValueError("Split manifest contains patients absent from data")
    pd.DataFrame(inventory).to_csv(out / "patient_file_hashes.csv", index=False)
    result = {"patient_counts": dict(counts), "patients": len(inventory),
              "hours": sum(r["n_hours"] for r in inventory), "all_schema_valid": True,
              "contiguous_hourly_rows": True, "duplicate_full_files": 0,
              "cohort_sha256": digest.hexdigest(), "manifest_matches_data": mapping is not None,
              "scope": "Exact-byte duplicates and ID disjointness; cannot establish undocumented subject linkage"}
    (out / "DATA_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--output-dir", default="reports/week1")
    p.add_argument("--manifest")
    args = p.parse_args()
    manifest = pd.read_csv(args.manifest) if args.manifest else None
    print(audit_data(args.data_root, args.output_dir, manifest))


if __name__ == "__main__":
    main()

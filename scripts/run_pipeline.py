"""Single resumable entrypoint for the implemented SepsisGuard data pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

import pandas as pd

from sepsisguard.audit import audit_data
from sepsisguard.download import run_download
from sepsisguard.eda import run_eda
from sepsisguard.preprocessing import TrainOnlyPreprocessor, manifest_digest
from sepsisguard.schema import EXPECTED_PATIENT_COUNTS
from sepsisguard.splits import check_split_integrity, make_splits
from sepsisguard.synthetic import make_demo
from scripts.compute_check import detect_runtime
from scripts.run_week2 import run as run_preprocessing

LEGACY_RUN = "20261021T183143Z-82ed151a"
FORMAT = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_fingerprint(project):
    project = Path(project)
    paths = [project / "pyproject.toml"]
    paths += sorted((project / "src/sepsisguard").glob("*.py"))
    paths += sorted((project / "scripts").glob("*.py"))
    h = hashlib.sha256()
    for path in paths:
        h.update(path.relative_to(project).as_posix().encode() + b"\0")
        h.update(bytes.fromhex(digest(path)))
    return h.hexdigest()


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", value):
        raise ValueError("Invalid run ID")
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def snapshot_root(root, state):
    relative = Path(state.get("snapshot", "."))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe checkpoint snapshot path")
    return Path(root) / relative


def verify_checkpoint(root, state):
    root = snapshot_root(root, state)
    if state.get("format") != FORMAT or not state.get("artifacts"):
        raise ValueError("Unsupported or empty pipeline checkpoint")
    for relative, expected in state["artifacts"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe checkpoint artifact path")
        target = Path(root) / path
        if not target.is_file() or digest(target) != expected:
            raise ValueError(f"Checkpoint integrity failure: {relative}")


def choose_run(project, storage, requested, config):
    if requested != "auto":
        return safe_id(requested)
    candidates = {}
    roots = [Path(project) / "runs"]
    if storage is not None:
        roots.append(Path(storage) / "runs")
    for root in roots:
        for path in root.glob("pipeline-*/checkpoint.json"):
            state = json.loads(path.read_text())
            if state.get("format") == FORMAT and state.get("config") == config:
                candidates[state["run_id"]] = state.get("created_at", "")
    if candidates:
        return safe_id(max(candidates, key=candidates.get))
    return "pipeline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]


def checkpoint(run_dir, storage, state, stage):
    artifacts = {}
    for directory in (run_dir / "data", run_dir / "preprocessing"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.name.endswith(".partial"):
                artifacts[path.relative_to(run_dir).as_posix()] = digest(path)
    for name in ("PIPELINE_RESULT.json", "PIPELINE_REPORT.md"):
        if (run_dir / name).exists():
            artifacts[name] = digest(run_dir / name)
    updated = {**state, "stage": stage, "artifacts": artifacts}
    if storage is not None:
        destination = Path(storage) / "runs" / state["run_id"]
        relative = "snapshots/" + uuid.uuid4().hex
        generation = destination / relative
        generation.mkdir(parents=True)
        # Save immutable generations. An interrupted copy leaves the previous
        # completion pointer and its files intact and available for restoration.
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.name != "checkpoint.json" and not path.name.endswith(".partial"):
                saved = generation / path.relative_to(run_dir)
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, saved)
                if digest(saved) != digest(path):
                    raise ValueError(f"Persistent file readback failed: {saved}")
        verify_checkpoint(generation, updated)
        write_json(generation / "checkpoint.json", updated)
        pointer = {**updated, "snapshot": relative}
        write_json(destination / "checkpoint.json", pointer)
        assert json.loads((destination / "checkpoint.json").read_text()) == pointer
    write_json(run_dir / "checkpoint.json", updated)
    return updated


def import_legacy(storage, run_id, target):
    if storage is None or not run_id:
        return None
    safe_id(run_id)
    saved = Path(storage) / "reports" / run_id / "week1"
    result_path = Path(storage) / "runs" / run_id / "WEEK1_RESULT.json"
    if not saved.exists() and not result_path.exists():
        return None
    required = ("DATA_AUDIT.json", "split_manifest.csv", "patient_summary.csv", "EDA_REPORT.md")
    if not result_path.is_file() or not all((saved / name).is_file() for name in required):
        raise ValueError("Existing imported checkpoint is incomplete; restore it before continuing")
    result = json.loads(result_path.read_text())
    audit = json.loads((saved / "DATA_AUDIT.json").read_text())
    if result.get("run_id") != run_id or result.get("status") != "week1_data_pipeline_passed" or result.get("audit") != audit:
        raise ValueError("Imported milestone and data audit disagree")
    shutil.copytree(saved, target, dirs_exist_ok=True)
    # Preserve historical evidence, but remove calendar labels from the active report.
    report = target / "EDA_REPORT.md"
    report.write_text(report.read_text().replace("# Week 1 Exploratory Data Analysis", "# Exploratory Data Analysis"))
    return {"run_id": run_id, "cohort_sha256": audit["cohort_sha256"],
            "manifest_file_sha256": digest(target / "split_manifest.csv")}


def command_log(command, path, project, *, echo=True):
    with path.open("w") as log:
        process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            if echo:
                print(line, end="", flush=True)
            log.write(line)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


def run_pipeline(project, data_root, storage_root=None, *, run_id="auto",
                 import_run_id=LEGACY_RUN, mode="real", run_checks=True):
    project, data_root = Path(project).resolve(), Path(data_root).resolve()
    storage = Path(storage_root).resolve() if storage_root else None
    if mode not in ("real", "synthetic"):
        raise ValueError("Unknown execution mode")
    if mode == "real" and not run_checks:
        raise ValueError("Real-data pipeline requires the regression and leakage tests")
    config = {"mode": mode, "seed": 42, "min_observed_fraction": 0.0,
              "source_fingerprint": source_fingerprint(project)}
    run_id = choose_run(project, storage, run_id, config)
    run_dir = project / "runs" / run_id
    if storage is not None and (storage == run_dir or run_dir in storage.parents):
        raise ValueError("Persistent storage must be outside the working run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "checkpoint.json"
    saved = storage / "runs" / run_id if storage else None
    if saved is not None and (saved == run_dir or run_dir in saved.parents or saved in run_dir.parents):
        raise ValueError("Persistent and working checkpoint directories must not overlap")
    local_valid = False
    if state_path.exists():
        local_state = json.loads(state_path.read_text())
        if local_state.get("config") != config or local_state.get("run_id") != run_id:
            raise ValueError("Run configuration changed; use a new run ID to preserve prior results")
        try:
            verify_checkpoint(run_dir, local_state)
            local_valid = True
        except ValueError:
            if saved is None or not (saved / "checkpoint.json").is_file():
                raise
    if not local_valid and saved is not None and (saved / "checkpoint.json").is_file():
        restored = json.loads((saved / "checkpoint.json").read_text())
        if restored.get("config") != config or restored.get("run_id") != run_id:
            raise ValueError("Checkpoint configuration differs; use a new run ID")
        verify_checkpoint(saved, restored)
        shutil.copytree(snapshot_root(saved, restored), run_dir, dirs_exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "format": FORMAT, "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "stage": "started", "imported_checkpoint": None}
    if state.get("config") != config or state.get("run_id") != run_id:
        raise ValueError("Run configuration changed; use a new run ID to preserve prior results")
    if state["stage"] != "started":
        verify_checkpoint(run_dir, state)
    data_out, pre_out = run_dir / "data", run_dir / "preprocessing"
    data_out.mkdir(exist_ok=True)
    compute = json.loads(subprocess.check_output([sys.executable, str(project / "scripts/compute_check.py")], text=True))
    write_json(run_dir / "compute.json", compute)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    write_json(run_dir / "source.json", {"commit": commit, "fingerprint": config["source_fingerprint"],
                                         "mode": mode, "runtime": detect_runtime()})
    print(f"Run: {run_id}; runtime: {compute['runtime']}; mode: {mode}", flush=True)
    if compute["cuda_available"]:
        raise ValueError("Select a CPU runtime for data preparation")
    if run_checks:
        print("Running regression and leakage checks", flush=True)
        command_log([sys.executable, "-m", "pytest", "-q"], run_dir / "tests.log", project)
    command_log([sys.executable, "-m", "pip", "freeze"], run_dir / "environment.log", project, echo=False)
    if mode == "real":
        if (data_root / "SYNTHETIC_DATA.txt").exists():
            raise ValueError("Synthetic files cannot be used for a real-data run")
        archives = storage / "dataset-archives" if storage else project / "data/downloads"
        run_download(data_root, archives)
        expected = EXPECTED_PATIENT_COUNTS
    else:
        if not data_root.exists():
            make_demo(data_root, n_per_source=80)
        if not (data_root / "SYNTHETIC_DATA.txt").is_file():
            raise ValueError("Synthetic mode requires explicitly marked synthetic data")
        expected = {"A": 80, "B": 80}
    cached_data = state["stage"] in ("data_ready", "complete")
    if not cached_data:
        state["imported_checkpoint"] = import_legacy(storage, import_run_id, data_out) if mode == "real" else None
        if state["imported_checkpoint"] is None:
            print("Generating descriptive EDA and fixed patient partitions", flush=True)
            run_eda(data_root, data_out)
            make_splits(pd.read_csv(data_out / "patient_summary.csv"), seed=42).to_csv(data_out / "split_manifest.csv", index=False)
    else:
        print("Reusing saved EDA and patient partitions", flush=True)
    manifest_path = data_out / "split_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    check_split_integrity(manifest)
    previous = json.loads((data_out / "DATA_AUDIT.json").read_text()) if (data_out / "DATA_AUDIT.json").exists() else None
    print("Auditing all raw files against the saved patient partitions", flush=True)
    audit = audit_data(data_root, data_out, manifest, expected_counts=expected)
    if previous and audit != previous:
        raise ValueError("Cohort changed since the saved audit; use a new run after investigating")
    imported = state["imported_checkpoint"]
    if imported and (audit["cohort_sha256"] != imported["cohort_sha256"] or digest(manifest_path) != imported["manifest_file_sha256"]):
        raise ValueError("Imported data or patient partition changed")
    input_key = {"cohort_sha256": audit["cohort_sha256"], "manifest_sha256": manifest_digest(manifest)}
    reuse_pre = cached_data and state["stage"] == "complete" and state.get("input") == input_key
    state["input"] = input_key
    if not reuse_pre:
        # Incomplete preprocessing artifacts are not a completed fit and must be replaced.
        if pre_out.exists():
            shutil.rmtree(pre_out)
        state = checkpoint(run_dir, storage, state, "data_ready")
        print("Fitting causal preprocessing using only A/train", flush=True)
        result = run_preprocessing(data_root, manifest_path, pre_out, min_observed_fraction=0.0)
        write_json(pre_out / "PREPROCESSING_RESULT.json", result)
        (pre_out / "WEEK2_RESULT.json").unlink(missing_ok=True)
    else:
        print("Reusing the verified preprocessing checkpoint", flush=True)
    result = json.loads((pre_out / "PREPROCESSING_RESULT.json").read_text())
    pre = TrainOnlyPreprocessor.load(pre_out / "preprocessor.json")
    if (result.get("status") != "preprocessing_and_dataset_checks_passed"
            or result["test_B_opened"] is not False
            or pre.provenance["manifest_sha256"] != input_key["manifest_sha256"]):
        raise ValueError("Preprocessing evidence does not match the locked partition")
    summary = {
        "status": "data_pipeline_passed" if mode == "real" else "synthetic_pipeline_passed",
        "run_id": run_id, "mode": mode, "runtime": compute["runtime"], "source_commit": commit,
        "source_fingerprint": config["source_fingerprint"], "tests": "passed" if run_checks else "not_run",
        "audit": audit, "input": input_key, "imported_checkpoint": imported,
        "preprocessing": result, "preprocessing_reused": reuse_pre,
        "persistent_checkpoint": str(saved) if saved else None,
        "scope": "Data download, EDA, patient splits, audit, leakage tests, causal preprocessing and Dataset checks",
        "predictive_models_trained": False, "independent_holdout_evaluated": False,
    }
    write_json(run_dir / "PIPELINE_RESULT.json", summary)
    (run_dir / "PIPELINE_REPORT.md").write_text(
        "# SepsisGuard-UAR pipeline result\n\n" + ("**SYNTHETIC DEMO ONLY.**\n\n" if mode == "synthetic" else "")
        + f"Run: `{run_id}`\n\nStatus: {summary['status']}\n\n"
        + f"Patients: {audit['patients']:,}; hours: {audit['hours']:,}.\n\n"
        + f"Selected predictors: {len(pre.features)}; input channels: {result['input_channels']}; trailing hours: 12.\n\n"
        + "Preprocessing fits A/train only. B is not used for fitting or development windows.\n\n"
        + "No predictive model training, calibration, retrieval, or clinical validation is included in this result.\n")
    checkpoint(run_dir, storage, state, "complete")
    print(json.dumps(summary, indent=2), flush=True)
    print("Completed pipeline; " + ("persistent checkpoint readback verified: " + str(saved) if saved else "local checkpoint only: " + str(run_dir)), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--storage-root", help="Mounted private Drive project folder")
    p.add_argument("--run-id", default="auto", help="Resume the latest matching pipeline run, or choose a new ID")
    p.add_argument("--import-run-id", default=LEGACY_RUN, help="Import an existing audited milestone if present")
    p.add_argument("--demo", action="store_true", help="Synthetic smoke test only; never a research result")
    a = p.parse_args()
    run_pipeline(Path.cwd(), a.data_root, a.storage_root, run_id=a.run_id,
                 import_run_id=a.import_run_id, mode="synthetic" if a.demo else "real")


if __name__ == "__main__":
    main()

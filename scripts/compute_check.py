"""Colab Pro / Windows / Linux compute inspection (no GPU required)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def detect_runtime() -> str:
    """Recognize Colab in a fresh subprocess before google.colab is imported."""
    if not Path("/content").exists():
        return "local_or_server"
    if any(os.environ.get(key) for key in ("COLAB_RELEASE_TAG", "COLAB_BACKEND_VERSION")):
        return "colab"
    try:
        return "colab" if importlib.util.find_spec("google.colab") is not None else "local_or_server"
    except ModuleNotFoundError:
        return "local_or_server"


def inspect_compute() -> dict:
    report = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "processor": platform.processor() or "unknown",
        "runtime": detect_runtime(),
        "cpu_count": os.cpu_count(),
        "local_disk_free_gb": round(shutil.disk_usage("/content" if Path("/content").exists() else ".").free / 2**30, 2),
        "cuda_available": False,
        "gpus": [],
    }
    try:
        import psutil
        report["memory_total_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
        report["memory_available_gb"] = round(psutil.virtual_memory().available / 2**30, 2)
    except ImportError:
        report["memory_total_gb"] = "unknown (optional: pip install psutil)"
    try:
        import torch
        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                report["gpus"].append({
                    "name": torch.cuda.get_device_name(i),
                    "vram_gb": round(prop.total_memory / 2**30, 2),
                    "compute_capability": f"{prop.major}.{prop.minor}",
                })
    except ImportError:
        report["torch_version"] = "not installed"
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Record compute details for reproducibility")
    p.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = p.parse_args()
    report = inspect_compute()
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

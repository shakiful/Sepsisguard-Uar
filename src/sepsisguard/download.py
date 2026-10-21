"""Get PhysioNet 2019 training archives; validate extracted record counts.

Official source: https://physionet.org/content/challenge-2019/1.0.0/
The two ZIP endpoints are linked in historical community code. If they are
unavailable, use the documented AWS S3 sync fallback from README instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from urllib.request import urlopen, Request
from zipfile import ZipFile, BadZipFile
from tempfile import TemporaryDirectory
import subprocess
import sys

from .schema import EXPECTED_PATIENT_COUNTS

BASE = "https://physionet.org/files/challenge-2019/1.0.0/"


def download_archive(source: str, path: Path) -> None:
    url = f"{BASE}training_set{source}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    req = Request(url, headers={"User-Agent": "SepsisGuard-UAR/0.1 (academic reproducibility)"})
    try:
        with urlopen(req, timeout=90) as response, tmp.open("wb") as f:
            shutil.copyfileobj(response, f, length=2**20)
        with ZipFile(tmp) as z:
            if not any(n.lower().endswith(".psv") for n in z.namelist()):
                raise ValueError(f"Downloaded {url} but archive has no PSV files")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def extract_archive(path: Path, target: Path) -> int:
    """Only extract PSV files by basename, ignoring ZIP's internal nesting."""
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(path) as archive:
        entries = [i for i in archive.infolist() if not i.is_dir() and i.filename.lower().endswith(".psv")]
        names = [Path(i.filename).name for i in entries]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate patient names inside {path}")
        for info, name in zip(entries, names):
            destination = target / name
            if destination.exists():
                continue
            with archive.open(info) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return len(list(target.glob("*.psv")))


def run_download(data_root: str | Path, archives_dir: str | Path, transport: str = "auto") -> None:
    root = Path(data_root)
    cache = Path(archives_dir)
    if transport not in ("auto", "zip", "s3"):
        raise ValueError("transport must be auto, zip, or s3")
    for source, expected in EXPECTED_PATIENT_COUNTS.items():
        target = root / f"training_set{source}"
        found = len(list(target.glob("*.psv"))) if target.exists() else 0
        if found == expected:
            print(f"Dataset {source}: already extracted {found:,} patients")
            continue
        archive = cache / f"training_set{source}.zip"
        if not archive.exists() and transport != "s3":
            print(f"Downloading source {source} archive to {archive}")
            try:
                download_archive(source, archive)
            except Exception as e:
                if transport == "zip":
                    raise RuntimeError(f"ZIP unavailable: {e}") from e
                print(f"Historical ZIP unavailable ({e}); using official anonymous S3 mirror", flush=True)
        if archive.exists() and transport != "s3":
            try:
                count = extract_archive(archive, target)
            except (BadZipFile, ValueError) as e:
                raise RuntimeError(f"Invalid ZIP {archive}: {e}. Rebuild cache using --transport s3") from e
        else:
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, "-m", "awscli", "s3", "sync", "--no-sign-request",
                f"s3://physionet-open/challenge-2019/1.0.0/training/training_set{source}/",
                str(target), "--only-show-errors"], check=True)
            count = len(list(target.glob("*.psv")))
        if count != expected:
            raise RuntimeError(
                f"Dataset {source} contains {count:,} PSV files, expected {expected:,}. "
                "Download may be incomplete; do not run publishable EDA yet.")
        print(f"Dataset {source}: validated {count:,} patient records")
        if not archive.exists():
            # Preserve official bytes in a local cache, atomically, for Colab resets.
            from zipfile import ZIP_DEFLATED
            cache.mkdir(parents=True, exist_ok=True)
            partial = archive.with_suffix(".partial")
            with ZipFile(partial, "w", compression=ZIP_DEFLATED, compresslevel=3) as z:
                for path in sorted(target.glob("*.psv")):
                    z.write(path, arcname=path.name)
            partial.replace(archive)


def main() -> None:
    p = argparse.ArgumentParser(description="Download/extract official PhysioNet 2019 ZIP files")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--archives-dir", default="data/downloads",
                   help="Point at persistent Google Drive storage on Colab")
    p.add_argument("--transport", choices=("auto", "zip", "s3"), default="auto")
    a = p.parse_args()
    run_download(a.data_root, a.archives_dir, a.transport)

if __name__ == "__main__":
    main()

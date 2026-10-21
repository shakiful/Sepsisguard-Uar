"""Create obviously synthetic demo files for end-to-end smoke testing (NOT research!)."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from .schema import FEATURES, LABEL


def make_demo(root: str | Path, n_per_source: int = 80, seed: int = 7) -> None:
    if n_per_source < 50:
        raise ValueError("Use >=50 demo patients per source for split validation")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "SYNTHETIC_DATA.txt").write_text("Artificial smoke-test records. Never report these as research results.\n")
    rng = np.random.default_rng(seed)
    for source in ("A", "B"):
        folder = root / f"training_set{source}"
        folder.mkdir(parents=True, exist_ok=True)
        for idx in range(n_per_source):
            n = 12 + (idx % 9)
            df = pd.DataFrame(np.nan, index=range(n), columns=FEATURES, dtype=float)
            for col, base in (("HR", 80), ("O2Sat", 97), ("Temp", 37), ("MAP", 79), ("Resp", 17), ("WBC", 8)):
                vals = base + rng.normal(0, 1, size=n)
                vals[rng.random(n) < (0.45 if col == "WBC" else 0.1)] = np.nan
                df[col] = vals
            df["Age"] = 45 + (idx % 40)
            df["Gender"] = idx % 2
            df["Unit1"] = idx % 2
            df["Unit2"] = 1 - (idx % 2)
            df["HospAdmTime"] = -8
            df["ICULOS"] = np.arange(1, n + 1)
            # This is a deterministic artificial label, not a clinical model.
            labels = np.zeros(n, dtype=int)
            if idx % 4 == 0:
                labels[n - 4:] = 1
            df[LABEL] = labels
            df.to_csv(folder / f"p{idx:06d}.psv", sep="|", index=False, na_rep="NaN")
    print(f"SYNTHETIC DEMO ONLY: wrote {n_per_source * 2} fake patient files under {root}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/synthetic")
    p.add_argument("--n-per-source", type=int, default=80)
    args = p.parse_args()
    make_demo(args.output, args.n_per_source)

if __name__ == "__main__":
    main()

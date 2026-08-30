"""Download the IEEE-CIS competition data from Kaggle.

Uses the Kaggle Python client and the stdlib zipfile module rather than
shelling out to `unzip`, so it behaves the same on Windows, macOS and Linux.

Prerequisites:
  1. kaggle.com -> Account -> Create New API Token (downloads kaggle.json)
  2. Place kaggle.json at ~/.kaggle/kaggle.json  (Windows: %USERPROFILE%\\.kaggle\\)
  3. Accept the rules at https://www.kaggle.com/c/ieee-fraud-detection
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudlens.config import DATA_DIR

COMPETITION = "ieee-fraud-detection"
EXPECTED = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
]


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if all((DATA_DIR / f).exists() for f in EXPECTED):
        print(f"All four CSVs already present in {DATA_DIR}; nothing to do.")
        return 0

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("The kaggle package is missing. Install it with: pip install kaggle")
        return 1

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as exc:
        print(f"Kaggle authentication failed: {exc}")
        print("Confirm kaggle.json is in place, then retry.")
        return 1

    print(f"Downloading '{COMPETITION}' into {DATA_DIR} (about 1.2 GB) ...")
    try:
        api.competition_download_files(COMPETITION, path=str(DATA_DIR), quiet=False)
    except Exception as exc:
        print(f"Download failed: {exc}")
        print("Have you accepted the competition rules on the Kaggle website?")
        return 1

    archive = DATA_DIR / f"{COMPETITION}.zip"
    if archive.exists():
        print("Extracting ...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(DATA_DIR)
        archive.unlink()

    missing = [f for f in EXPECTED if not (DATA_DIR / f).exists()]
    if missing:
        print(f"Extraction incomplete, still missing: {missing}")
        return 1

    print("Done. Files:")
    for f in EXPECTED:
        print(f"  {f}  ({(DATA_DIR / f).stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

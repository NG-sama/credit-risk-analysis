"""Download the HMEQ dataset from Kaggle into data/raw/hmeq.csv.

Kaggle's Python client only reads KAGGLE_USERNAME / KAGGLE_KEY from the
environment (checked at import time), but this repo's .env uses
KAGGLE_API_KEY, so we map it before importing kaggle.
"""

import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

DATASET = "ajay1735/hmeq-data"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
TARGET_CSV = RAW_DIR / "hmeq.csv"

MANUAL_DOWNLOAD_MSG = (
    "Kaggle CLI download failed. Manually download the CSV from "
    "https://www.kaggle.com/datasets/ajay1735/hmeq-data/data and place it "
    f"at {TARGET_CSV}, then re-run."
)


def _prepare_kaggle_env() -> None:
    load_dotenv()
    if "KAGGLE_KEY" not in os.environ and "KAGGLE_API_KEY" in os.environ:
        os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_KEY"]


def _find_downloaded_csv() -> Path | None:
    csvs = list(RAW_DIR.glob("*.csv"))
    return csvs[0] if csvs else None


def download() -> Path:
    if TARGET_CSV.exists():
        print(f"{TARGET_CSV} already exists, skipping download.")
        return TARGET_CSV

    _prepare_kaggle_env()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                DATASET,
                "-p",
                str(RAW_DIR),
                "--unzip",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(MANUAL_DOWNLOAD_MSG) from exc

    downloaded = _find_downloaded_csv()
    if downloaded is None:
        raise RuntimeError(MANUAL_DOWNLOAD_MSG)
    if downloaded != TARGET_CSV:
        shutil.move(str(downloaded), str(TARGET_CSV))

    print(f"Downloaded dataset to {TARGET_CSV}")
    return TARGET_CSV


if __name__ == "__main__":
    download()

'''
This script downloads and extracts all of the Distnet scenarios (datasets) used for the project.
'''

from pathlib import Path

import urllib.request
import zipfile

DATA_URL = "http://www.automl.org/wp-content/uploads/2018/04/DistNetData.zip"
DATA_ROOT = Path("data")
DISTNET_DATA_DIR = DATA_ROOT / "distnet_data"
ZIP_PATH = DATA_ROOT / "DistNetData.zip"


def download_and_extract_data():
    """Downloads and extracts the dataset if it doesn't already exist."""
    print("--- Checking Data ---")

    # Check if directory exists and has contents
    if DISTNET_DATA_DIR.exists() and any(DISTNET_DATA_DIR.iterdir()):
        print(f"Data already found in {DISTNET_DATA_DIR}. Skipping download.")
        return

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Downloading data from {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

    print("Extracting data...")
    # Extract directly to data/, which will yield data/DistNetData based on the zip's internal structure
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_ROOT)

    extracted_folder = DATA_ROOT / "DistNetData"

    if extracted_folder.exists():
        # If the target distnet_data folder was created empty, remove it so we can rename
        if DISTNET_DATA_DIR.exists():
            DISTNET_DATA_DIR.rmdir()
        # Rename DistNetData -> distnet_data
        extracted_folder.rename(DISTNET_DATA_DIR)

    print("Cleaning up zip file...")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    print(
        "✅Data download and extraction complete. Target folder is correctly structured."
    )


if __name__ == "__main__":
    download_and_extract_data()

'''
This module defines the common paths used across the project.
'''

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DISTNET_DATA_DIR = ROOT_DIR / "data" / "distnet_data"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
EXPERIMENTS_DATA_DIR = ROOT_DIR / "experiments_data"
RESULTS_DIR = ROOT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

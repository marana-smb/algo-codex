"""
Centralized project paths.

This module defines standard directories used across notebooks,
experiments, and utilities so that file paths are never hard-coded.

Example usage:

    from paths import DATA_RAW, OUTPUTS

    df = pd.read_csv(DATA_RAW / "features.csv")
    df.to_csv(OUTPUTS / "model_output.csv", index=False)

"""

from pathlib import Path

# -------------------------------------------------------
# Project root
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -------------------------------------------------------
# Data directories
# -------------------------------------------------------

DATA = PROJECT_ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERMEDIATE = DATA / "intermediate"

# -------------------------------------------------------
# Output directories
# -------------------------------------------------------

OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"
TABLES = OUTPUTS / "tables"

# -------------------------------------------------------
# Code / research directories
# -------------------------------------------------------

NOTEBOOKS = PROJECT_ROOT / "notebooks"
EXPERIMENTS = PROJECT_ROOT / "experiments"

# -------------------------------------------------------
# Ensure key directories exist
# -------------------------------------------------------

_REQUIRED_DIRS = [
    DATA,
    DATA_RAW,
    DATA_INTERMEDIATE,
    OUTPUTS,
    FIGURES,
    MODELS,
    TABLES,
]

for directory in _REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# Utility helpers
# -------------------------------------------------------

def output_path(filename: str) -> Path:
    """
    Return path inside the outputs directory.

    Example:
        df.to_csv(output_path("feature_check.csv"))
    """
    return OUTPUTS / filename


def model_path(filename: str) -> Path:
    """
    Return path for saved model artifacts.
    """
    return MODELS / filename


def figure_path(filename: str) -> Path:
    """
    Return path for saved figures.
    """
    return FIGURES / filename
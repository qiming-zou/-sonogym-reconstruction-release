"""Patient split utilities for reconstruction-only experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "trajectory_generation/data_splits/reconstruction_patients.json"


def parse_patient_ids(patient_ids: str | None) -> list[str] | None:
    if not patient_ids:
        return None
    ids = [patient_id.strip() for patient_id in patient_ids.split(",") if patient_id.strip()]
    return ids or None


def load_patient_split(split_file: str | os.PathLike[str], split: str) -> list[str]:
    with Path(split_file).open("r") as f:
        data = json.load(f)
    if split not in data:
        raise KeyError(f"Split `{split}` was not found in {split_file}.")
    patient_ids = [str(patient_id) for patient_id in data[split]]
    if not patient_ids:
        raise ValueError(f"Split `{split}` in {split_file} is empty.")
    return patient_ids


def resolve_patient_ids(
    patient_ids: str | None,
    split: str | None,
    split_file: str | os.PathLike[str] = DEFAULT_SPLIT_FILE,
) -> list[str] | None:
    explicit_ids = parse_patient_ids(patient_ids)
    if explicit_ids is not None:
        return explicit_ids
    if split:
        return load_patient_split(split_file, split)
    return None


def apply_patient_ids_env(patient_ids: list[str] | None) -> None:
    if patient_ids:
        os.environ["SONOGYM_PATIENT_IDS"] = ",".join(patient_ids)

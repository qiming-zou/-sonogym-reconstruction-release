"""Build a target-anatomy 3D prior volume from patient label maps."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from ruamel.yaml import YAML


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = (
    PROJECT_ROOT
    / "source/spinal_surgery/spinal_surgery/tasks/robot_US_reconstruction/cfgs/robotic_US_reconstruction.yaml"
)
DEFAULT_ASSET_ROOT = PROJECT_ROOT / "source/spinal_surgery/spinal_surgery/assets/data/HumanModels"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from trajectory_generation.patient_splits import DEFAULT_SPLIT_FILE, resolve_patient_ids


parser = argparse.ArgumentParser(description="Build an anatomical target prior from 3D label maps.")
parser.add_argument("--config", type=str, default=str(DEFAULT_CFG))
parser.add_argument("--asset_root", type=str, default=str(DEFAULT_ASSET_ROOT))
parser.add_argument("--output", type=str, default="artifacts/checkpoints/anatomy_prior_l4.pt")
parser.add_argument("--target_label", type=int, default=None)
parser.add_argument("--surface_only", action="store_true", default=True)
parser.add_argument("--full_volume", action="store_true", help="Use full target mask instead of target surface.")
parser.add_argument("--split_file", type=str, default=str(DEFAULT_SPLIT_FILE))
parser.add_argument("--split", type=str, default=None, help="Patient split name, for example train or test.")
parser.add_argument("--patient_ids", type=str, default=None, help="Comma-separated patient id override.")
args = parser.parse_args()


def target_label_from_name(name: str) -> int:
    return 32 - int(str(name)[-1])


def surface_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    interior = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return center & ~interior


def sample_aligned_mask(mask: np.ndarray, center: np.ndarray, volume_size: list[int], scale: float) -> np.ndarray:
    grids = np.meshgrid(
        *[np.arange(size, dtype=np.float32) for size in volume_size],
        indexing="ij",
    )
    coords = np.stack(grids, axis=-1)
    coords = center.reshape(1, 1, 1, 3) + (coords - (np.asarray(volume_size, dtype=np.float32) / 2.0)) * scale
    coords = np.rint(coords).astype(np.int64)
    valid = np.ones(volume_size, dtype=bool)
    for axis in range(3):
        valid &= (coords[..., axis] >= 0) & (coords[..., axis] < mask.shape[axis])
        coords[..., axis] = np.clip(coords[..., axis], 0, mask.shape[axis] - 1)
    sampled = mask[coords[..., 0], coords[..., 1], coords[..., 2]]
    sampled[~valid] = False
    return sampled.astype(np.float32)


def main() -> None:
    with Path(args.config).open("r") as f:
        cfg = YAML().load(f)
    patient_ids = resolve_patient_ids(args.patient_ids, args.split, args.split_file)
    if patient_ids is None:
        patient_ids = [str(patient_id) for patient_id in cfg["patient"]["id_list"]]
    target_name = str(cfg["reconstruction"]["target_vertebra"])
    target_label = args.target_label if args.target_label is not None else target_label_from_name(target_name)
    label_res = float(cfg["patient"]["label_res"])
    volume_res = float(cfg["reconstruction"]["volume_res"])
    volume_size = [int(value) for value in cfg["reconstruction"]["volume_size"]]
    label_voxel_step = volume_res / label_res

    aligned = []
    centers = {}
    source_files = {}
    for patient_id in patient_ids:
        label_path = Path(args.asset_root) / "selected_dataset_stl" / patient_id / "combined_label_map.nii.gz"
        label_map = nib.load(str(label_path)).get_fdata().astype(np.int16)
        mask = label_map == int(target_label)
        if not np.any(mask):
            raise ValueError(f"Target label {target_label} was not found in {label_path}.")
        coords = np.argwhere(mask)
        center = coords.mean(axis=0).astype(np.float32)
        train_mask = mask if args.full_volume else surface_mask(mask)
        aligned.append(sample_aligned_mask(train_mask, center, volume_size, label_voxel_step))
        centers[patient_id] = center.tolist()
        source_files[patient_id] = str(label_path)

    stack = np.stack(aligned, axis=0)
    prior_volume = torch.from_numpy(stack.mean(axis=0).astype(np.float32))
    output = {
        "kind": "anatomy_prior_volume",
        "method": "aligned_target_surface_probability",
        "split": args.split,
        "split_file": str(args.split_file),
        "target_anatomy": target_name,
        "target_label": int(target_label),
        "patient_ids": patient_ids,
        "num_patients": len(patient_ids),
        "volume_size": volume_size,
        "volume_res": volume_res,
        "label_res": label_res,
        "label_voxel_step": float(label_voxel_step),
        "surface_only": not args.full_volume,
        "prior_volume": prior_volume,
        "centers_label_voxels": centers,
        "source_files": source_files,
        "stats": {
            "min": float(prior_volume.min().item()),
            "max": float(prior_volume.max().item()),
            "mean": float(prior_volume.mean().item()),
            "sum": float(prior_volume.sum().item()),
            "nonzero_voxels": int((prior_volume > 0).sum().item()),
        },
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(output, args.output)
    print(f"[RESULT] saved anatomy prior to {args.output}")
    print(f"[RESULT] target={target_name} label={target_label}")
    print(f"[RESULT] num_patients={len(patient_ids)}")
    print(f"[RESULT] prior_shape={tuple(prior_volume.shape)}")
    print(f"[RESULT] prior_max={output['stats']['max']:.6f}")
    print(f"[RESULT] prior_mean={output['stats']['mean']:.6f}")
    print(f"[RESULT] prior_sum={output['stats']['sum']:.6f}")


if __name__ == "__main__":
    main()

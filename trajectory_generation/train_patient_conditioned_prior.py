"""Train a patient-conditioned 3D anatomy prior from label maps.

The model learns p(target volume | sparse early reconstruction, global prior).
Synthetic sparse reconstructions are generated from patient-specific target
surfaces so the same network can consume online ``human_rec_volume`` at test
time.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ruamel.yaml import YAML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajectory_generation.patient_conditioned_prior import (  # noqa: E402
    PatientConditionedPriorNet,
    build_prior_input,
    dice_loss_from_logits,
)
from trajectory_generation.patient_splits import DEFAULT_SPLIT_FILE, load_patient_split  # noqa: E402

DEFAULT_CFG = (
    PROJECT_ROOT
    / "source/spinal_surgery/spinal_surgery/tasks/robot_US_reconstruction/cfgs/robotic_US_reconstruction.yaml"
)
DEFAULT_ASSET_ROOT = PROJECT_ROOT / "source/spinal_surgery/spinal_surgery/assets/data/HumanModels"


parser = argparse.ArgumentParser(description="Train patient-conditioned anatomy prior.")
parser.add_argument("--config", type=str, default=str(DEFAULT_CFG))
parser.add_argument("--asset_root", type=str, default=str(DEFAULT_ASSET_ROOT))
parser.add_argument("--split_file", type=str, default=str(DEFAULT_SPLIT_FILE))
parser.add_argument("--train_split", type=str, default="train")
parser.add_argument("--val_split", type=str, default="test")
parser.add_argument("--output", type=str, default="artifacts/checkpoints/patient_conditioned_prior_l4.pt")
parser.add_argument("--target_label", type=int, default=None)
parser.add_argument("--epochs", type=int, default=160)
parser.add_argument("--samples_per_patient", type=int, default=48)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--base_channels", type=int, default=12)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--bce_weight", type=float, default=0.45)
parser.add_argument("--dice_weight", type=float, default=0.55)
parser.add_argument("--threshold", type=float, default=0.35)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")


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


def load_target_volume(
    patient_id: str,
    cfg: dict,
    asset_root: Path,
    target_label: int,
) -> tuple[torch.Tensor, np.ndarray, tuple[int, int, int]]:
    label_path = asset_root / "selected_dataset_stl" / patient_id / "combined_label_map.nii.gz"
    label_map = nib.load(str(label_path)).get_fdata().astype(np.int16)
    mask = label_map == int(target_label)
    if not np.any(mask):
        raise ValueError(f"Target label {target_label} was not found in {label_path}.")
    coords = np.argwhere(mask)
    center = coords.mean(axis=0).astype(np.float32)
    label_voxel_step = float(cfg["reconstruction"]["volume_res"]) / float(cfg["patient"]["label_res"])
    volume_size = [int(value) for value in cfg["reconstruction"]["volume_size"]]
    target = sample_aligned_mask(surface_mask(mask), center, volume_size, label_voxel_step)
    return torch.from_numpy(target.astype(np.float32)), center, tuple(int(value) for value in label_map.shape)


def load_geometry_volume(
    patient_id: str,
    cfg: dict,
    asset_root: Path,
    center: np.ndarray,
    label_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Encode patient surface geometry on the reconstruction grid.

    Channels are signed y-distance to the skin surface and the local surface
    normal.  The target label is not used here, only the precomputed body
    surface maps that the simulator already uses for probe contact.
    """
    patient_dir = asset_root / "selected_dataset_stl" / patient_id
    surface_path = patient_dir / "body_lowest_y_array.pt"
    normal_path = patient_dir / "body_surface_normal_array.pt"
    volume_size = [int(value) for value in cfg["reconstruction"]["volume_size"]]
    label_voxel_step = float(cfg["reconstruction"]["volume_res"]) / float(cfg["patient"]["label_res"])

    if not surface_path.exists() or not normal_path.exists():
        return torch.zeros((4, *volume_size), dtype=torch.float32)

    surface_y = torch.load(surface_path, map_location="cpu").float()
    normals = torch.load(normal_path, map_location="cpu").float()
    normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)

    x_coords = torch.arange(volume_size[0], dtype=torch.float32)
    y_coords = torch.arange(volume_size[1], dtype=torch.float32)
    z_coords = torch.arange(volume_size[2], dtype=torch.float32)
    label_x = torch.round(torch.as_tensor(float(center[0])) + (x_coords - volume_size[0] / 2.0) * label_voxel_step).long()
    label_y = torch.as_tensor(float(center[1])) + (y_coords - volume_size[1] / 2.0) * label_voxel_step
    label_z = torch.round(torch.as_tensor(float(center[2])) + (z_coords - volume_size[2] / 2.0) * label_voxel_step).long()

    valid_x = (label_x >= 0) & (label_x < label_shape[0])
    valid_z = (label_z >= 0) & (label_z < label_shape[2])
    label_x = label_x.clamp(0, surface_y.shape[0] - 1)
    label_z = label_z.clamp(0, surface_y.shape[1] - 1)

    sampled_surface_y = surface_y[label_x[:, None], label_z[None, :]]
    valid_surface = (sampled_surface_y >= 0) & valid_x[:, None] & valid_z[None, :]
    sampled_normals = normals[label_x[:, None], label_z[None, :], :]

    distance_scale = max(1.0, float(volume_size[1]) * label_voxel_step * 0.5)
    signed_surface_distance = (
        (label_y[None, :, None] - sampled_surface_y[:, None, :]) / distance_scale
    ).clamp(-2.0, 2.0) * 0.5
    signed_surface_distance = torch.where(
        valid_surface[:, None, :],
        signed_surface_distance,
        torch.zeros_like(signed_surface_distance),
    )

    geometry_channels = [signed_surface_distance]
    for normal_channel in range(3):
        channel = sampled_normals[..., normal_channel][:, None, :].expand(*volume_size)
        channel = torch.where(valid_surface[:, None, :], channel, torch.zeros_like(channel))
        geometry_channels.append(channel)
    return torch.stack(geometry_channels, dim=0).float()


def make_sparse_observation(target: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Generate a plausible early sparse reconstruction from a target volume."""
    sparse = torch.zeros_like(target)
    coords = target.nonzero(as_tuple=False)
    if coords.numel() == 0:
        return sparse

    mode = int(torch.randint(0, 4, (1,), generator=generator).item())
    num_points = coords.shape[0]
    if mode == 0:
        keep_prob = float((0.02 + 0.10 * torch.rand((), generator=generator)).item())
        keep = torch.rand((num_points,), generator=generator) < keep_prob
        sparse[coords[keep, 0], coords[keep, 1], coords[keep, 2]] = 1.0
    elif mode == 1:
        center = coords[int(torch.randint(0, num_points, (1,), generator=generator).item())]
        radius = float((4.0 + 6.0 * torch.rand((), generator=generator)).item())
        dist = torch.linalg.norm((coords - center).float(), dim=1)
        keep = dist <= radius
        sparse[coords[keep, 0], coords[keep, 1], coords[keep, 2]] = 1.0
    elif mode == 2:
        axis = int(torch.randint(0, 3, (1,), generator=generator).item())
        center = int(coords[int(torch.randint(0, num_points, (1,), generator=generator).item()), axis].item())
        width = int(torch.randint(2, 7, (1,), generator=generator).item())
        keep = (coords[:, axis] >= center - width) & (coords[:, axis] <= center + width)
        sparse[coords[keep, 0], coords[keep, 1], coords[keep, 2]] = 1.0
    else:
        current = coords[int(torch.randint(0, num_points, (1,), generator=generator).item())].float()
        steps = int(torch.randint(3, 8, (1,), generator=generator).item())
        for _ in range(steps):
            radius = float((3.0 + 4.0 * torch.rand((), generator=generator)).item())
            dist = torch.linalg.norm(coords.float() - current, dim=1)
            keep = dist <= radius
            sparse[coords[keep, 0], coords[keep, 1], coords[keep, 2]] = 1.0
            current = current + torch.randn((3,), generator=generator) * 4.0
            current = current.clamp(0.0, float(target.shape[0] - 1))

    noise_prob = float((0.004 * torch.rand((), generator=generator)).item())
    if noise_prob > 0:
        sparse = torch.maximum(sparse, (torch.rand(target.shape, generator=generator) < noise_prob).float())
    return sparse


class SyntheticPriorDataset(Dataset):
    def __init__(
        self,
        patient_ids: list[str],
        target_volumes: dict[str, torch.Tensor],
        geometry_volumes: dict[str, torch.Tensor],
        samples_per_patient: int,
        seed: int,
    ):
        self.patient_ids = list(patient_ids)
        self.target_volumes = target_volumes
        self.geometry_volumes = geometry_volumes
        self.samples_per_patient = int(samples_per_patient)
        self.seed = int(seed)
        self.items = [
            (patient_id, sample_index)
            for patient_id in self.patient_ids
            for sample_index in range(self.samples_per_patient)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        patient_id, sample_index = self.items[index]
        target = self.target_volumes[patient_id]
        geometry = self.geometry_volumes[patient_id]
        generator = torch.Generator().manual_seed(self.seed + index * 9973 + sample_index)
        sparse = make_sparse_observation(target, generator)
        return sparse, geometry, target, patient_id


def evaluate(
    model: PatientConditionedPriorNet,
    loader: DataLoader,
    global_prior: torch.Tensor,
    threshold: float,
    device: torch.device,
) -> dict:
    model.eval()
    dices = []
    recalls = []
    precisions = []
    with torch.inference_mode():
        for sparse, geometry, target, _patient_id in loader:
            sparse = sparse.to(device)
            geometry = geometry.to(device)
            target = target.to(device)
            logits = model(build_prior_input(sparse, global_prior, geometry))
            prob = torch.sigmoid(logits).squeeze(1)
            pred = prob >= threshold
            tgt = target > 0.5
            inter = (pred & tgt).sum(dim=(1, 2, 3)).float()
            pred_sum = pred.sum(dim=(1, 2, 3)).float().clamp_min(1.0)
            tgt_sum = tgt.sum(dim=(1, 2, 3)).float().clamp_min(1.0)
            dices.extend(((2.0 * inter) / (pred_sum + tgt_sum)).detach().cpu().tolist())
            recalls.extend((inter / tgt_sum).detach().cpu().tolist())
            precisions.extend((inter / pred_sum).detach().cpu().tolist())
    return {
        "dice": float(np.mean(dices)) if dices else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "precision": float(np.mean(precisions)) if precisions else 0.0,
    }


def main() -> None:
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    with Path(args.config).open("r") as f:
        cfg = YAML().load(f)
    target_name = str(cfg["reconstruction"]["target_vertebra"])
    target_label = args.target_label if args.target_label is not None else target_label_from_name(target_name)
    train_patients = load_patient_split(args.split_file, args.train_split)
    val_patients = load_patient_split(args.split_file, args.val_split)
    all_patients = sorted(set(train_patients + val_patients))

    target_data = {
        patient_id: load_target_volume(patient_id, cfg, Path(args.asset_root), int(target_label))
        for patient_id in all_patients
    }
    target_volumes = {patient_id: data[0] for patient_id, data in target_data.items()}
    geometry_volumes = {
        patient_id: load_geometry_volume(patient_id, cfg, Path(args.asset_root), data[1], data[2])
        for patient_id, data in target_data.items()
    }
    global_prior = torch.stack([target_volumes[patient_id] for patient_id in train_patients], dim=0).mean(dim=0)

    train_dataset = SyntheticPriorDataset(
        train_patients,
        target_volumes,
        geometry_volumes,
        args.samples_per_patient,
        args.seed,
    )
    val_dataset = SyntheticPriorDataset(
        val_patients,
        target_volumes,
        geometry_volumes,
        max(8, args.samples_per_patient // 3),
        args.seed + 100000,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_in_channels = 9
    model = PatientConditionedPriorNet(in_channels=model_in_channels, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = ((1.0 - global_prior).sum() / global_prior.sum().clamp_min(1.0)).clamp(1.0, 300.0).to(device)

    best_state = None
    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for sparse, geometry, target, _patient_id in train_loader:
            sparse = sparse.to(device)
            geometry = geometry.to(device)
            target = target.to(device).unsqueeze(1)
            logits = model(build_prior_input(sparse, global_prior.to(device), geometry))
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            dice = dice_loss_from_logits(logits, target)
            loss = args.bce_weight * bce + args.dice_weight * dice
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            train_metrics = evaluate(model, train_loader, global_prior, args.threshold, device)
            val_metrics = evaluate(model, val_loader, global_prior, args.threshold, device)
            score = val_metrics["dice"] + 0.5 * val_metrics["recall"]
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )
            print(
                f"[EPOCH {epoch:04d}] loss={np.mean(losses):.4f} "
                f"train_dice={train_metrics['dice']:.4f} val_dice={val_metrics['dice']:.4f} "
                f"val_recall={val_metrics['recall']:.4f} val_precision={val_metrics['precision']:.4f}"
            )
            if score > best_score:
                best_score = score
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    output = {
        "kind": "patient_conditioned_anatomy_prior",
        "method": "surface_geometry_conditioned_sparse_reconstruction_3d_unet",
        "target_anatomy": target_name,
        "target_label": int(target_label),
        "train_patients": train_patients,
        "val_patients": val_patients,
        "volume_size": [int(value) for value in cfg["reconstruction"]["volume_size"]],
        "volume_res": float(cfg["reconstruction"]["volume_res"]),
        "label_res": float(cfg["patient"]["label_res"]),
        "prior_volume": global_prior.detach().cpu(),
        "prior_model_state": best_state,
        "model_config": {"in_channels": model_in_channels, "base_channels": int(args.base_channels)},
        "geometry_channels": ["signed_skin_y_distance", "surface_normal_x", "surface_normal_y", "surface_normal_z"],
        "geometry_by_patient": {key: value.detach().cpu() for key, value in geometry_volumes.items()},
        "threshold": float(args.threshold),
        "history": history,
        "training": vars(args),
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(output, args.output)
    print(f"[RESULT] saved patient-conditioned prior to {args.output}")
    print(f"[RESULT] best_score={best_score:.6f}")


if __name__ == "__main__":
    main()

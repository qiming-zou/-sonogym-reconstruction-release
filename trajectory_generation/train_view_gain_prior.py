"""Train a view-conditioned gain model for ultrasound pose selection."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ruamel.yaml import YAML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajectory_generation.patient_splits import DEFAULT_SPLIT_FILE, load_patient_split  # noqa: E402
from trajectory_generation.patient_conditioned_prior import build_prior_input  # noqa: E402
from trajectory_generation.train_patient_conditioned_prior import (  # noqa: E402
    DEFAULT_ASSET_ROOT,
    DEFAULT_CFG,
    load_geometry_volume,
    load_target_volume,
    make_sparse_observation,
    target_label_from_name,
)
from trajectory_generation.view_gain_prior import ViewGainNet, build_view_gain_input  # noqa: E402


parser = argparse.ArgumentParser(description="Train candidate-view coverage-gain predictor.")
parser.add_argument("--config", type=str, default=str(DEFAULT_CFG))
parser.add_argument("--asset_root", type=str, default=str(DEFAULT_ASSET_ROOT))
parser.add_argument("--split_file", type=str, default=str(DEFAULT_SPLIT_FILE))
parser.add_argument("--train_split", type=str, default="train")
parser.add_argument("--val_split", type=str, default="test")
parser.add_argument("--output", type=str, default="artifacts/checkpoints/view_gain_prior_l4.pt")
parser.add_argument("--target_label", type=int, default=None)
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--samples_per_patient", type=int, default=32)
parser.add_argument("--candidates_per_sparse", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--base_channels", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--half_length", type=float, default=12.0)
parser.add_argument("--half_width", type=float, default=4.0)
parser.add_argument("--thickness", type=float, default=1.25)
parser.add_argument("--roll_span", type=float, default=0.4)
parser.add_argument("--target_center_prob", type=float, default=0.7)
parser.add_argument("--seed", type=int, default=13)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")


def make_plane_mask(
    volume_shape: tuple[int, int, int],
    center: torch.Tensor,
    yaw: torch.Tensor,
    roll: torch.Tensor,
    half_length: float,
    half_width: float,
    thickness: float,
    coords: torch.Tensor | None = None,
) -> torch.Tensor:
    if coords is None:
        axes = [torch.arange(size, dtype=torch.float32) for size in volume_shape]
        x, y, z = torch.meshgrid(*axes, indexing="ij")
        coords = torch.stack([x, y, z], dim=-1)
    direction = torch.stack([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw)])
    vertical = torch.tensor([0.0, 1.0, 0.0])
    horizontal_perp = torch.stack([-torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)])
    second_axis = vertical * torch.cos(roll) + horizontal_perp * torch.sin(roll)
    normal = torch.cross(direction, second_axis, dim=0)
    normal = normal / normal.norm().clamp_min(1e-6)
    rel = coords - center.reshape(1, 1, 1, 3)
    along = (rel * direction.reshape(1, 1, 1, 3)).sum(dim=-1)
    across = (rel * second_axis.reshape(1, 1, 1, 3)).sum(dim=-1)
    distance = (rel * normal.reshape(1, 1, 1, 3)).sum(dim=-1).abs()
    return ((along.abs() <= half_length) & (across.abs() <= half_width) & (distance <= thickness)).float()


def sample_candidate_center(
    target: torch.Tensor,
    generator: torch.Generator,
    target_center_prob: float,
) -> torch.Tensor:
    target_coords = target.nonzero(as_tuple=False).float()
    if target_coords.numel() > 0 and torch.rand((), generator=generator).item() < target_center_prob:
        center = target_coords[int(torch.randint(0, target_coords.shape[0], (1,), generator=generator).item())]
        jitter = torch.randn((3,), generator=generator) * torch.tensor([4.0, 3.0, 4.0])
        center = center + jitter
    else:
        shape = torch.tensor(target.shape, dtype=torch.float32)
        center = torch.rand((3,), generator=generator) * (shape - 1.0)
    max_corner = torch.tensor(target.shape, dtype=torch.float32) - 1.0
    return torch.minimum(torch.maximum(center, torch.zeros_like(center)), max_corner)


class SyntheticViewGainDataset(Dataset):
    def __init__(
        self,
        patient_ids: list[str],
        target_volumes: dict[str, torch.Tensor],
        geometry_volumes: dict[str, torch.Tensor],
        samples_per_patient: int,
        candidates_per_sparse: int,
        seed: int,
        half_length: float,
        half_width: float,
        thickness: float,
        roll_span: float,
        target_center_prob: float,
    ):
        self.patient_ids = list(patient_ids)
        self.target_volumes = target_volumes
        self.geometry_volumes = geometry_volumes
        self.samples_per_patient = int(samples_per_patient)
        self.candidates_per_sparse = int(candidates_per_sparse)
        self.seed = int(seed)
        self.half_length = float(half_length)
        self.half_width = float(half_width)
        self.thickness = float(thickness)
        self.roll_span = float(roll_span)
        self.target_center_prob = float(target_center_prob)
        volume_shape = tuple(next(iter(target_volumes.values())).shape)
        axes = [torch.arange(size, dtype=torch.float32) for size in volume_shape]
        x, y, z = torch.meshgrid(*axes, indexing="ij")
        self.coords = torch.stack([x, y, z], dim=-1)
        self.items = [
            (patient_id, sparse_index, candidate_index)
            for patient_id in self.patient_ids
            for sparse_index in range(self.samples_per_patient)
            for candidate_index in range(self.candidates_per_sparse)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        patient_id, sparse_index, candidate_index = self.items[index]
        target = self.target_volumes[patient_id]
        geometry = self.geometry_volumes[patient_id]
        generator = torch.Generator().manual_seed(self.seed + sparse_index * 100003 + candidate_index * 9973 + index)
        sparse = make_sparse_observation(target, generator)
        center = sample_candidate_center(target, generator, self.target_center_prob)
        yaw = torch.rand((), generator=generator) * torch.pi
        roll = (torch.rand((), generator=generator) * 2.0 - 1.0) * self.roll_span
        plane = make_plane_mask(
            tuple(target.shape),
            center,
            yaw,
            roll,
            self.half_length,
            self.half_width,
            self.thickness,
            self.coords,
        )
        uncovered_target = target.float() * (1.0 - sparse.float().clamp(0.0, 1.0))
        gain = (uncovered_target * plane).sum() / target.sum().clamp_min(1.0)
        return sparse.float(), geometry.float(), plane.float(), gain.float(), patient_id


def evaluate(
    model: ViewGainNet,
    loader: DataLoader,
    global_prior: torch.Tensor,
    device: torch.device,
) -> dict:
    model.eval()
    preds = []
    targets = []
    losses = []
    with torch.inference_mode():
        for sparse, geometry, plane, gain, _patient_id in loader:
            sparse = sparse.to(device)
            geometry = geometry.to(device)
            plane = plane.to(device)
            gain = gain.to(device)
            pred = torch.nn.functional.softplus(
                model(build_view_gain_input(sparse, global_prior.to(device), plane, geometry))
            )
            loss = torch.nn.functional.smooth_l1_loss(pred, gain)
            preds.append(pred.detach().cpu())
            targets.append(gain.detach().cpu())
            losses.append(float(loss.detach().cpu()))
    pred_all = torch.cat(preds) if preds else torch.zeros(0)
    target_all = torch.cat(targets) if targets else torch.zeros(0)
    if pred_all.numel() > 1 and target_all.std() > 1e-8 and pred_all.std() > 1e-8:
        corr = float(torch.corrcoef(torch.stack([pred_all, target_all]))[0, 1].item())
    else:
        corr = 0.0
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "mae": float((pred_all - target_all).abs().mean().item()) if preds else 0.0,
        "corr": corr,
        "target_mean": float(target_all.mean().item()) if targets else 0.0,
        "pred_mean": float(pred_all.mean().item()) if preds else 0.0,
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

    train_dataset = SyntheticViewGainDataset(
        train_patients,
        target_volumes,
        geometry_volumes,
        args.samples_per_patient,
        args.candidates_per_sparse,
        args.seed,
        args.half_length,
        args.half_width,
        args.thickness,
        args.roll_span,
        args.target_center_prob,
    )
    val_dataset = SyntheticViewGainDataset(
        val_patients,
        target_volumes,
        geometry_volumes,
        max(8, args.samples_per_patient // 3),
        args.candidates_per_sparse,
        args.seed + 100000,
        args.half_length,
        args.half_width,
        args.thickness,
        args.roll_span,
        args.target_center_prob,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ViewGainNet(in_channels=10, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_score = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for sparse, geometry, plane, gain, _patient_id in train_loader:
            sparse = sparse.to(device)
            geometry = geometry.to(device)
            plane = plane.to(device)
            gain = gain.to(device)
            pred = torch.nn.functional.softplus(
                model(build_view_gain_input(sparse, global_prior.to(device), plane, geometry))
            )
            loss = torch.nn.functional.smooth_l1_loss(pred, gain)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            train_metrics = evaluate(model, train_loader, global_prior, device)
            val_metrics = evaluate(model, val_loader, global_prior, device)
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )
            print(
                f"[EPOCH {epoch:04d}] loss={np.mean(losses):.6f} "
                f"train_mae={train_metrics['mae']:.6f} val_mae={val_metrics['mae']:.6f} "
                f"val_corr={val_metrics['corr']:.4f}"
            )
            if val_metrics["mae"] < best_score:
                best_score = val_metrics["mae"]
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    output = {
        "kind": "view_conditioned_gain_prior",
        "method": "synthetic_sparse_reconstruction_candidate_plane_3d_cnn",
        "target_anatomy": target_name,
        "target_label": int(target_label),
        "train_patients": train_patients,
        "val_patients": val_patients,
        "volume_size": [int(value) for value in cfg["reconstruction"]["volume_size"]],
        "volume_res": float(cfg["reconstruction"]["volume_res"]),
        "label_res": float(cfg["patient"]["label_res"]),
        "prior_volume": global_prior.detach().cpu(),
        "view_gain_model_state": best_state,
        "model_config": {"in_channels": 10, "base_channels": int(args.base_channels)},
        "geometry_channels": ["signed_skin_y_distance", "surface_normal_x", "surface_normal_y", "surface_normal_z"],
        "geometry_by_patient": {key: value.detach().cpu() for key, value in geometry_volumes.items()},
        "history": history,
        "training": vars(args),
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(output, args.output)
    print(f"[RESULT] saved view-gain prior to {args.output}")
    print(f"[RESULT] best_val_mae={best_score:.6f}")


if __name__ == "__main__":
    main()

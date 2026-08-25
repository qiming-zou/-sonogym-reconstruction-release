"""Patient-conditioned anatomy-prior model and online inference utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatientConditionedPriorNet(nn.Module):
    """Small 3D UNet for target-volume belief from sparse online reconstruction."""

    def __init__(self, in_channels: int = 5, base_channels: int = 12):
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        c = self.base_channels
        self.enc0 = ConvBlock3d(self.in_channels, c)
        self.enc1 = ConvBlock3d(c, c * 2)
        self.enc2 = ConvBlock3d(c * 2, c * 4)
        self.mid = ConvBlock3d(c * 4, c * 4)
        self.up1 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3d(c * 4, c * 2)
        self.up0 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec0 = ConvBlock3d(c * 2, c)
        self.out = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.enc0(x)
        x1 = self.enc1(F.avg_pool3d(x0, kernel_size=2))
        x2 = self.enc2(F.avg_pool3d(x1, kernel_size=2))
        xm = self.mid(x2)
        y1 = self.up1(xm)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))
        y0 = self.up0(y1)
        y0 = self.dec0(torch.cat([y0, x0], dim=1))
        return self.out(y0)


def coordinate_channels(
    volume_shape: tuple[int, int, int],
    device: torch.device | str,
    batch_size: int = 1,
) -> torch.Tensor:
    axes = [torch.linspace(-1.0, 1.0, steps=size, device=device) for size in volume_shape]
    x, y, z = torch.meshgrid(*axes, indexing="ij")
    coords = torch.stack([x, y, z], dim=0).unsqueeze(0)
    return coords.repeat(batch_size, 1, 1, 1, 1)


def build_prior_input(
    sparse_reconstruction: torch.Tensor,
    global_prior: torch.Tensor,
    geometry: torch.Tensor | None = None,
) -> torch.Tensor:
    if sparse_reconstruction.ndim != 4:
        raise ValueError(f"Expected sparse reconstruction shape (B, X, Y, Z), got {tuple(sparse_reconstruction.shape)}")
    batch_size = sparse_reconstruction.shape[0]
    device = sparse_reconstruction.device
    prior = global_prior.to(device).float().unsqueeze(0).repeat(batch_size, 1, 1, 1)
    sparse = sparse_reconstruction.float().clamp(0.0, 1.0)
    coords = coordinate_channels(tuple(sparse.shape[-3:]), device, batch_size)
    channels = [sparse.unsqueeze(1), prior.unsqueeze(1), coords]
    if geometry is not None:
        if geometry.ndim != 5:
            raise ValueError(f"Expected geometry shape (B, C, X, Y, Z), got {tuple(geometry.shape)}")
        if geometry.shape[0] != batch_size or tuple(geometry.shape[-3:]) != tuple(sparse.shape[-3:]):
            raise ValueError(
                "Geometry must match sparse reconstruction batch and volume shape: "
                f"geometry={tuple(geometry.shape)}, sparse={tuple(sparse.shape)}"
            )
        channels.append(geometry.to(device).float())
    return torch.cat(channels, dim=1)


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.float()
    reduce_dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=reduce_dims)
    denom = prob.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


@dataclass
class PatientPriorPredictor:
    model: PatientConditionedPriorNet
    global_prior: torch.Tensor
    threshold: float
    geometry_by_patient: dict[str, torch.Tensor] | None = None

    def _geometry_for_patients(
        self,
        patient_ids: list[str] | None,
        sparse_reconstruction: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor | None:
        expected_geometry_channels = max(0, int(self.model.in_channels) - 5)
        if expected_geometry_channels == 0:
            return None

        shape = tuple(sparse_reconstruction.shape[-3:])
        if patient_ids is None:
            return torch.zeros(
                (sparse_reconstruction.shape[0], expected_geometry_channels, *shape),
                dtype=torch.float32,
                device=device,
            )
        if len(patient_ids) != sparse_reconstruction.shape[0]:
            raise ValueError(
                f"Expected {sparse_reconstruction.shape[0]} patient ids, got {len(patient_ids)}."
            )

        geometry_by_patient = self.geometry_by_patient or {}
        geometry = []
        for patient_id in patient_ids:
            patient_geometry = geometry_by_patient.get(str(patient_id))
            if patient_geometry is None:
                patient_geometry = torch.zeros((expected_geometry_channels, *shape), dtype=torch.float32)
            if patient_geometry.shape[0] != expected_geometry_channels or tuple(patient_geometry.shape[-3:]) != shape:
                raise ValueError(
                    f"Geometry for {patient_id} has shape {tuple(patient_geometry.shape)}, "
                    f"expected ({expected_geometry_channels}, {shape[0]}, {shape[1]}, {shape[2]})."
                )
            geometry.append(patient_geometry.to(device).float())
        return torch.stack(geometry, dim=0)

    @torch.inference_mode()
    def predict(self, sparse_reconstruction: torch.Tensor, patient_ids: list[str] | None = None) -> torch.Tensor:
        self.model.eval()
        device = next(self.model.parameters()).device
        sparse_reconstruction = sparse_reconstruction.to(device)
        geometry = self._geometry_for_patients(patient_ids, sparse_reconstruction, device)
        model_input = build_prior_input(sparse_reconstruction, self.global_prior.to(device), geometry)
        return torch.sigmoid(self.model(model_input)).squeeze(1)


def load_patient_prior_predictor(
    checkpoint_path: str,
    device: torch.device | str,
    threshold: float | None = None,
) -> PatientPriorPredictor:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "prior_model_state" not in checkpoint:
        raise KeyError(f"{checkpoint_path} does not contain `prior_model_state`.")
    model_cfg = checkpoint.get("model_config", {})
    model = PatientConditionedPriorNet(
        int(model_cfg.get("in_channels", 5)),
        int(model_cfg.get("base_channels", 12)),
    ).to(device)
    model.load_state_dict(checkpoint["prior_model_state"])
    global_prior = checkpoint["prior_volume"].float().to(device)
    geometry_by_patient = checkpoint.get("geometry_by_patient")
    if geometry_by_patient is not None:
        geometry_by_patient = {str(key): value.float().cpu() for key, value in geometry_by_patient.items()}
    return PatientPriorPredictor(
        model=model,
        global_prior=global_prior,
        threshold=float(checkpoint.get("threshold", 0.35) if threshold is None else threshold),
        geometry_by_patient=geometry_by_patient,
    )

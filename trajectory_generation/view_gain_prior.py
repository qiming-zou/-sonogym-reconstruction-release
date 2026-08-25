"""View-conditioned coverage-gain model for ultrasound pose planning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from trajectory_generation.patient_conditioned_prior import ConvBlock3d, coordinate_channels


class ViewGainNet(nn.Module):
    """Predict expected target coverage gain for a candidate ultrasound plane."""

    def __init__(self, in_channels: int = 10, base_channels: int = 8):
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        c = self.base_channels
        self.enc0 = ConvBlock3d(self.in_channels, c)
        self.enc1 = ConvBlock3d(c, c * 2)
        self.enc2 = ConvBlock3d(c * 2, c * 4)
        self.head = nn.Sequential(
            nn.Linear(c * 4, c * 2),
            nn.SiLU(inplace=True),
            nn.Linear(c * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc0(x)
        x = self.enc1(F.avg_pool3d(x, kernel_size=2))
        x = self.enc2(F.avg_pool3d(x, kernel_size=2))
        x = x.mean(dim=(2, 3, 4))
        return self.head(x).squeeze(-1)


def build_view_gain_input(
    sparse_reconstruction: torch.Tensor,
    prior_volume: torch.Tensor,
    plane_mask: torch.Tensor,
    geometry: torch.Tensor | None = None,
) -> torch.Tensor:
    if sparse_reconstruction.ndim != 4:
        raise ValueError(
            f"Expected sparse reconstruction shape (B, X, Y, Z), got {tuple(sparse_reconstruction.shape)}"
        )
    if plane_mask.ndim != 4:
        raise ValueError(f"Expected plane mask shape (B, X, Y, Z), got {tuple(plane_mask.shape)}")
    if tuple(plane_mask.shape) != tuple(sparse_reconstruction.shape):
        raise ValueError(
            "Plane mask must match sparse reconstruction shape: "
            f"plane={tuple(plane_mask.shape)}, sparse={tuple(sparse_reconstruction.shape)}"
        )
    batch_size = sparse_reconstruction.shape[0]
    device = sparse_reconstruction.device
    if prior_volume.ndim == 3:
        prior = prior_volume.to(device).float().unsqueeze(0).repeat(batch_size, 1, 1, 1)
    elif prior_volume.ndim == 4 and prior_volume.shape[0] == batch_size:
        prior = prior_volume.to(device).float()
    else:
        raise ValueError(f"Expected prior shape (X, Y, Z) or (B, X, Y, Z), got {tuple(prior_volume.shape)}")

    sparse = sparse_reconstruction.float().clamp(0.0, 1.0)
    plane = plane_mask.float().clamp(0.0, 1.0)
    coords = coordinate_channels(tuple(sparse.shape[-3:]), device, batch_size)
    channels = [sparse.unsqueeze(1), prior.unsqueeze(1), plane.unsqueeze(1), coords]
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


@dataclass
class ViewGainPredictor:
    model: ViewGainNet
    global_prior: torch.Tensor
    geometry_by_patient: dict[str, torch.Tensor] | None = None

    def _geometry_for_patients(
        self,
        patient_ids: list[str] | None,
        batch_size: int,
        volume_shape: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor | None:
        expected_geometry_channels = max(0, int(self.model.in_channels) - 6)
        if expected_geometry_channels == 0:
            return None
        if patient_ids is None:
            return torch.zeros((batch_size, expected_geometry_channels, *volume_shape), device=device)
        if len(patient_ids) != batch_size:
            raise ValueError(f"Expected {batch_size} patient ids, got {len(patient_ids)}.")
        geometry_by_patient = self.geometry_by_patient or {}
        geometry = []
        for patient_id in patient_ids:
            patient_geometry = geometry_by_patient.get(str(patient_id))
            if patient_geometry is None:
                patient_geometry = torch.zeros((expected_geometry_channels, *volume_shape), dtype=torch.float32)
            geometry.append(patient_geometry.to(device).float())
        return torch.stack(geometry, dim=0)

    @torch.inference_mode()
    def predict(
        self,
        sparse_reconstruction: torch.Tensor,
        plane_masks: torch.Tensor,
        patient_ids: list[str] | None = None,
        prior_volume: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.model.eval()
        device = next(self.model.parameters()).device
        sparse_reconstruction = sparse_reconstruction.to(device).float()
        plane_masks = plane_masks.to(device).float()
        if plane_masks.ndim == 4:
            plane_masks = plane_masks.unsqueeze(0)
        if plane_masks.ndim != 5:
            raise ValueError(f"Expected plane masks shape (B, K, X, Y, Z), got {tuple(plane_masks.shape)}")
        batch_size, num_candidates = plane_masks.shape[:2]
        if sparse_reconstruction.shape[0] != batch_size:
            raise ValueError(
                f"Sparse reconstruction batch {sparse_reconstruction.shape[0]} does not match plane batch {batch_size}."
            )
        volume_shape = tuple(sparse_reconstruction.shape[-3:])
        geometry = self._geometry_for_patients(patient_ids, batch_size, volume_shape, device)
        if geometry is not None:
            geometry = geometry[:, None].repeat(1, num_candidates, 1, 1, 1, 1).reshape(
                batch_size * num_candidates,
                geometry.shape[1],
                *volume_shape,
            )
        sparse = sparse_reconstruction[:, None].repeat(1, num_candidates, 1, 1, 1).reshape(
            batch_size * num_candidates,
            *volume_shape,
        )
        planes = plane_masks.reshape(batch_size * num_candidates, *volume_shape)
        prior = self.global_prior if prior_volume is None else prior_volume
        if prior.ndim == 4:
            prior = prior.to(device)[:, None].repeat(1, num_candidates, 1, 1, 1).reshape(
                batch_size * num_candidates,
                *volume_shape,
            )
        logits = self.model(build_view_gain_input(sparse, prior.to(device), planes, geometry))
        return F.softplus(logits).reshape(batch_size, num_candidates)


def load_view_gain_predictor(checkpoint_path: str, device: torch.device | str) -> ViewGainPredictor:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "view_gain_model_state" not in checkpoint:
        raise KeyError(f"{checkpoint_path} does not contain `view_gain_model_state`.")
    model_cfg = checkpoint.get("model_config", {})
    model = ViewGainNet(
        int(model_cfg.get("in_channels", 10)),
        int(model_cfg.get("base_channels", 8)),
    ).to(device)
    model.load_state_dict(checkpoint["view_gain_model_state"])
    geometry_by_patient = checkpoint.get("geometry_by_patient")
    if geometry_by_patient is not None:
        geometry_by_patient = {str(key): value.float().cpu() for key, value in geometry_by_patient.items()}
    return ViewGainPredictor(
        model=model,
        global_prior=checkpoint["prior_volume"].float().to(device),
        geometry_by_patient=geometry_by_patient,
    )

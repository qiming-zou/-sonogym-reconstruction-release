"""Standalone runner for the current best registration-prior AUS-SLAM method.

This script does not import this repository's ``trajectory_generation`` package.
It uses only the installed SonoGym pip packages and contains the current best
algorithm implementation in this single file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


TRAIN_PATIENTS = ["s0004", "s0006", "s0010", "s0012", "s0014", "s0015", "s0024", "s0028"]
TEST_PATIENTS = ["s0029", "s0030", "s0034", "s0038"]


def _resolve_patient_ids(patient_ids: str | None, split: str | None) -> list[str]:
    if patient_ids:
        return [item.strip() for item in patient_ids.split(",") if item.strip()]
    if split == "train":
        return TRAIN_PATIENTS
    if split == "test":
        return TEST_PATIENTS
    raise ValueError("Pass --patient_ids or --split train/test.")


def _ensure_registration_deps(auto_install: bool) -> None:
    missing = []
    for module, package in [("open3d", "open3d"), ("probreg", "probreg"), ("sklearn", "scikit-learn")]:
        try:
            __import__(module)
        except Exception:
            missing.append(package)
    if missing and auto_install:
        command = [sys.executable, "-m", "pip", "install", *missing]
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
    elif missing:
        print(
            "[WARN] Missing registration dependencies: "
            + ", ".join(missing)
            + ". Open3D/probreg registration will be degraded.",
            flush=True,
        )


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
    grids = np.meshgrid(*[np.arange(size, dtype=np.float32) for size in volume_size], indexing="ij")
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


def build_anatomy_prior(output: Path, patients: list[str]) -> None:
    import nibabel as nib
    import torch
    from ruamel.yaml import YAML
    from sonogym_reconstruction_data import assets_data_dir
    from spinal_surgery import PACKAGE_DIR

    cfg_path = Path(PACKAGE_DIR) / "tasks/robot_US_reconstruction/cfgs/robotic_US_reconstruction.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = YAML().load(f)
    asset_root = Path(assets_data_dir()) / "HumanModels"
    target_name = str(cfg["reconstruction"]["target_vertebra"])
    target_label = target_label_from_name(target_name)
    label_res = float(cfg["patient"]["label_res"])
    volume_res = float(cfg["reconstruction"]["volume_res"])
    volume_size = [int(value) for value in cfg["reconstruction"]["volume_size"]]
    label_voxel_step = volume_res / label_res

    aligned = []
    source_files = {}
    for patient_id in patients:
        label_path = asset_root / "selected_dataset_stl" / patient_id / "combined_label_map.nii.gz"
        label_map = nib.load(str(label_path)).get_fdata().astype(np.int16)
        mask = label_map == int(target_label)
        if not np.any(mask):
            raise ValueError(f"Target label {target_label} was not found in {label_path}.")
        center = np.argwhere(mask).mean(axis=0).astype(np.float32)
        aligned.append(sample_aligned_mask(surface_mask(mask), center, volume_size, label_voxel_step))
        source_files[patient_id] = str(label_path)

    prior_volume = torch.from_numpy(np.stack(aligned, axis=0).mean(axis=0).astype(np.float32))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "anatomy_prior_volume",
            "method": "aligned_target_surface_probability",
            "split": "train",
            "target_anatomy": target_name,
            "target_label": int(target_label),
            "patient_ids": patients,
            "num_patients": len(patients),
            "volume_size": volume_size,
            "volume_res": volume_res,
            "label_res": label_res,
            "surface_only": True,
            "prior_volume": prior_volume,
            "source_files": source_files,
            "stats": {
                "min": float(prior_volume.min().item()),
                "max": float(prior_volume.max().item()),
                "mean": float(prior_volume.mean().item()),
                "sum": float(prior_volume.sum().item()),
            },
        },
        output,
    )
    print(f"[RESULT] built anatomy prior at {output}", flush=True)


def _subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    index = np.linspace(0, points.shape[0] - 1, num=max_points).round().astype(np.int64)
    return points[index]


def _principal_yaw(points: np.ndarray) -> float:
    centered = points[:, [0, 2]] - points[:, [0, 2]].mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    return float(np.arctan2(axis[1], axis[0]))


def _yaw_rotation(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64)


def _nearest_confidence(transformed_template: np.ndarray, observed: np.ndarray, inlier_radius: float) -> tuple[float, float]:
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return 0.0, float("inf")
    if transformed_template.shape[0] == 0 or observed.shape[0] == 0:
        return 0.0, float("inf")
    nn = NearestNeighbors(n_neighbors=1).fit(transformed_template)
    dist, _ = nn.kneighbors(observed, return_distance=True)
    dist = dist.reshape(-1)
    fitness = float((dist <= inlier_radius).mean())
    rmse = float(np.sqrt(np.mean(np.square(np.minimum(dist, inlier_radius * 3.0)))))
    return fitness, rmse


def _open3d_coarse_transform(source: np.ndarray, target: np.ndarray, voxel_size: float) -> np.ndarray | None:
    try:
        import open3d as o3d
    except Exception:
        return None
    if source.shape[0] < 20 or target.shape[0] < 20:
        return None

    def make_pcd(points: np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        down = pcd.voxel_down_sample(voxel_size)
        if len(down.points) < 8:
            return None, None
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
        )
        return down, fpfh

    source_down, source_feature = make_pcd(source)
    target_down, target_feature = make_pcd(target)
    if source_down is None or target_down is None:
        return None
    try:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_feature,
            target_feature,
            True,
            voxel_size * 2.0,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 2.0),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(20000, 0.999),
        )
    except Exception:
        return None
    transform = np.asarray(result.transformation, dtype=np.float64)
    return transform if np.isfinite(transform).all() else None


def _probreg_refine(source: np.ndarray, target: np.ndarray, maxiter: int) -> np.ndarray:
    try:
        from probreg import cpd
    except Exception:
        return source
    if source.shape[0] < 10 or target.shape[0] < 10:
        return source
    try:
        result = cpd.registration_cpd(
            source.astype(np.float64),
            target.astype(np.float64),
            tf_type_name="rigid",
            w=0.2,
            maxiter=maxiter,
            tol=1.0e-3,
            update_scale=False,
        )
        return result.transform(source.astype(np.float64))
    except Exception:
        return source


@dataclass
class RegistrationPriorPredictor:
    global_prior: object
    min_observed_points: int = 30
    max_template_points: int = 1200
    max_observed_points: int = 900
    prior_threshold: float = 0.05
    coarse_voxel_size: float = 3.0
    inlier_radius: float = 3.0
    max_blend_weight: float = 0.55
    cpd_maxiter: int = 25

    def __post_init__(self) -> None:
        import torch

        prior = self.global_prior.detach().float().cpu()
        self.global_prior = prior
        coords = (prior > self.prior_threshold).nonzero(as_tuple=False).float()
        if coords.numel() == 0:
            coords = prior.nonzero(as_tuple=False).float()
        weights = prior[coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()].float()
        if coords.shape[0] > self.max_template_points:
            order = torch.argsort(weights, descending=True)[: self.max_template_points]
            coords = coords[order]
            weights = weights[order]
        self.template_points = coords.numpy().astype(np.float64)
        weights = weights / weights.max().clamp_min(1.0e-6)
        self.template_weights = weights.numpy().astype(np.float32)
        self.template_centroid = self.template_points.mean(axis=0)
        self.template_yaw = _principal_yaw(self.template_points) if self.template_points.shape[0] >= 3 else 0.0
        self.last_blend_weights = None
        self.last_registration_stats: list[dict] = []

    def _initial_transform(self, observed: np.ndarray) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        yaw = _principal_yaw(observed) if observed.shape[0] >= 3 else self.template_yaw
        rotation = _yaw_rotation(yaw - self.template_yaw)
        transformed_centroid = self.template_centroid @ rotation.T
        transform[:3, :3] = rotation
        transform[:3, 3] = observed.mean(axis=0) - transformed_centroid
        return transform

    def _register(self, observed: np.ndarray) -> tuple[np.ndarray, float, dict]:
        source = _subsample_points(self.template_points, self.max_template_points)
        target = _subsample_points(observed, self.max_observed_points)
        coarse = _open3d_coarse_transform(source, target, self.coarse_voxel_size)
        if coarse is None:
            coarse = self._initial_transform(target)
        coarse_source = source @ coarse[:3, :3].T + coarse[:3, 3]
        refined_source = _probreg_refine(coarse_source, target, self.cpd_maxiter)
        fitness, rmse = _nearest_confidence(refined_source, target, self.inlier_radius)
        count_weight = min(1.0, max(0.0, (observed.shape[0] - self.min_observed_points) / 120.0))
        rmse_weight = max(0.0, 1.0 - rmse / (self.inlier_radius * 2.0)) if np.isfinite(rmse) else 0.0
        blend_weight = self.max_blend_weight * count_weight * fitness * rmse_weight
        full_coarse = self.template_points @ coarse[:3, :3].T + coarse[:3, 3]
        if refined_source.shape[0] == coarse_source.shape[0] and coarse_source.shape[0] > 0:
            full_registered = full_coarse + (refined_source - coarse_source).mean(axis=0)
        else:
            full_registered = full_coarse
        return full_registered, float(blend_weight), {
            "observed_points": int(observed.shape[0]),
            "fitness": float(fitness),
            "rmse": float(rmse),
            "blend_weight": float(blend_weight),
        }

    def _rasterize(self, points: np.ndarray, weights: np.ndarray, shape: tuple[int, int, int]):
        import torch

        volume = torch.zeros(shape, dtype=torch.float32)
        index = np.rint(points).astype(np.int64)
        valid = (
            (index[:, 0] >= 0)
            & (index[:, 0] < shape[0])
            & (index[:, 1] >= 0)
            & (index[:, 1] < shape[1])
            & (index[:, 2] >= 0)
            & (index[:, 2] < shape[2])
        )
        if not np.any(valid):
            return self.global_prior.clone()
        index = index[valid]
        valid_weights = torch.from_numpy(weights[valid]).float()
        volume[index[:, 0], index[:, 1], index[:, 2]] = torch.maximum(
            volume[index[:, 0], index[:, 1], index[:, 2]],
            valid_weights,
        )
        return torch.maximum(volume, self.global_prior * 0.15)

    def predict(self, sparse_reconstruction, patient_ids: list[str] | None = None):
        import torch

        sparse = sparse_reconstruction.detach().float().cpu()
        shape = tuple(int(v) for v in sparse.shape[-3:])
        priors = []
        blend_weights = []
        stats = []
        for env_index in range(sparse.shape[0]):
            observed = (sparse[env_index] > 0).nonzero(as_tuple=False).float().numpy().astype(np.float64)
            if observed.shape[0] < self.min_observed_points:
                priors.append(self.global_prior.clone())
                blend_weights.append(0.0)
                stats.append({"observed_points": int(observed.shape[0]), "fitness": 0.0, "rmse": float("inf"), "blend_weight": 0.0})
                continue
            registered, blend_weight, env_stats = self._register(observed)
            priors.append(self._rasterize(registered, self.template_weights, shape))
            blend_weights.append(blend_weight)
            stats.append(env_stats)
        self.last_blend_weights = torch.tensor(blend_weights, dtype=torch.float32)
        self.last_registration_stats = stats
        return torch.stack(priors, dim=0).to(sparse_reconstruction.device)

    def blend_weight(self, sparse_reconstruction, patient_ids: list[str] | None = None):
        import torch

        if self.last_blend_weights is None or self.last_blend_weights.shape[0] != sparse_reconstruction.shape[0]:
            return torch.zeros((sparse_reconstruction.shape[0],), dtype=torch.float32)
        return self.last_blend_weights.to(sparse_reconstruction.device)


class ActiveUSSLAMGoalPlanner:
    def __init__(self, task_env, prior_volume, prior_predictor=None):
        import torch

        self.task_env = task_env
        self.prior_predictor = prior_predictor
        self.replan_interval = 10
        self.reach_radius = 75.0
        self.observation_radius = 22.0
        self.coverage_weight = 1.0
        self.uncertainty_weight = 1.5
        self.frontier_weight = 2.0
        self.prior_weight = 0.6
        self.distance_weight = 0.018
        self.revisit_weight = 0.55
        self.yaw = 1.57
        self.yaw_gain_weight = 0.35
        self.yaw_strip_length = 35.0
        self.yaw_strip_width = 10.0
        self.roll = 0.0
        self.pose_score_samples = 9
        self.prior_update_interval = 20
        self.prior_threshold = 0.05
        self.cached_goal = None
        rec = task_env.surface_reconstructor
        self.device = task_env.sim.device
        self.volume_size = tuple(int(value) for value in rec.volume_size.detach().cpu().tolist())
        x_idx = torch.arange(self.volume_size[0], device=self.device)
        z_idx = torch.arange(self.volume_size[2], device=self.device)
        self.x_grid, self.z_grid = torch.meshgrid(x_idx, z_idx, indexing="ij")
        self.visit_counts = torch.zeros((task_env.scene.num_envs, self.volume_size[0], self.volume_size[2]), device=self.device)
        self.uncertainty = torch.ones_like(self.visit_counts)
        self.stagnation_steps = torch.zeros((task_env.scene.num_envs,), dtype=torch.long, device=self.device)
        self.last_covered_count = None
        self.prior_volume = self._normalize_prior_volume(prior_volume.float().to(self.device), task_env.scene.num_envs)
        self.prior_xz = self._project_prior_volume(self.prior_volume, task_env.scene.num_envs)
        self.base_prior_xz = self.prior_xz.clone()
        self.base_prior_volume = self.prior_volume.clone()
        self.kernel = torch.ones((1, 1, 3, 3), device=self.device)
        self.yaw_offsets = torch.linspace(-0.4, 0.4, steps=5, dtype=torch.float32, device=self.device)
        self.roll_offsets = torch.linspace(-0.35, 0.35, steps=5, dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self.cached_goal = None
        self.visit_counts.zero_()
        self.uncertainty.fill_(1.0)
        self.stagnation_steps.zero_()
        self.last_covered_count = None
        self.prior_xz = self.base_prior_xz.clone()
        self.prior_volume = self.base_prior_volume.clone()

    def _normalize_prior_volume(self, prior_volume, num_envs: int):
        if prior_volume.ndim == 3:
            volume = prior_volume.unsqueeze(0).repeat(num_envs, 1, 1, 1)
        elif prior_volume.ndim == 4 and prior_volume.shape[0] == num_envs:
            volume = prior_volume
        elif prior_volume.ndim == 4 and prior_volume.shape[0] == 1:
            volume = prior_volume.repeat(num_envs, 1, 1, 1)
        else:
            raise ValueError(f"Expected prior volume shape (X,Y,Z) or (B,X,Y,Z), got {tuple(prior_volume.shape)}")
        return volume.float() / volume.float().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)

    def _project_prior_volume(self, prior_volume, num_envs: int):
        if prior_volume.ndim == 3:
            prior_xz = prior_volume.float().sum(dim=1)
            return (prior_xz / prior_xz.amax().clamp_min(1e-6)).unsqueeze(0).repeat(num_envs, 1, 1)
        prior_xz = prior_volume.float().sum(dim=2)
        return prior_xz / prior_xz.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    def _patient_ids_for_envs(self) -> list[str]:
        rec = self.task_env.surface_reconstructor
        human_ids = [os.path.basename(path.rstrip("/")) for path in rec.human_list]
        env_to_human = rec.env_to_human_inds.detach().cpu().tolist()
        return [human_ids[int(index)] for index in env_to_human]

    def _maybe_update_patient_prior(self, step: int) -> None:
        if self.prior_predictor is None:
            return
        if step > 0 and step % self.prior_update_interval != 0:
            return
        rec_volume = self.task_env.surface_reconstructor.human_rec_volume.detach().float()
        predicted_prior = self.prior_predictor.predict(rec_volume, self._patient_ids_for_envs()).to(self.device)
        predicted_volume = self._normalize_prior_volume(predicted_prior, rec_volume.shape[0])
        predicted_xz = self._project_prior_volume(predicted_volume, rec_volume.shape[0])
        patient_weight = self.prior_predictor.blend_weight(rec_volume, self._patient_ids_for_envs()).to(self.device)
        patient_weight = patient_weight.float().clamp(0.0, 1.0).reshape(-1, 1, 1)
        blended_prior = (1.0 - patient_weight) * self.base_prior_xz + patient_weight * predicted_xz
        self.prior_xz = blended_prior / blended_prior.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        patient_weight_3d = patient_weight.reshape(-1, 1, 1, 1)
        blended_volume = (1.0 - patient_weight_3d) * self.base_prior_volume + patient_weight_3d * predicted_volume
        self.prior_volume = blended_volume / blended_volume.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)

    def _covered_xz_mask(self):
        return self.task_env.surface_reconstructor.human_rec_volume.float().amax(dim=2) > 0

    def _frontier_mask(self, covered, target_support):
        import torch.nn.functional as F

        covered_f = covered.float().unsqueeze(1)
        neighbor_count = F.conv2d(covered_f, self.kernel, padding=1).squeeze(1)
        return target_support & (~covered) & (neighbor_count > 0)

    def _cmd_grid(self):
        rec = self.task_env.surface_reconstructor
        corner = rec.human_rec_volume_corner
        cmd_x = (corner[:, 0:1, None] + self.x_grid.unsqueeze(0) * float(rec.volume_res)) / float(rec.label_res)
        cmd_z = (corner[:, 2:3, None] + self.z_grid.unsqueeze(0) * float(rec.volume_res)) / float(rec.label_res)
        return cmd_x, cmd_z

    def _pose_for_indices(self, x_index, z_index, information):
        import torch

        half_length = 0.5 * self.yaw_strip_length
        half_width = 0.5 * self.yaw_strip_width
        rec = self.task_env.surface_reconstructor
        cell_size = max(float(rec.volume_res) / max(float(rec.label_res), 1e-6), 1e-6)
        max_dx = max(1, int(round(half_length / cell_size)))
        max_dz = max(1, int(round(half_length / cell_size)))
        cmd_x, cmd_z = self._cmd_grid()
        volume_information = torch.where(rec.human_rec_volume.float() > 0, torch.zeros_like(self.prior_volume), self.prior_volume)
        sample_axis = torch.linspace(-1.0, 1.0, steps=self.pose_score_samples, dtype=torch.float32, device=self.device)
        sample_u, sample_v = torch.meshgrid(sample_axis, sample_axis, indexing="ij")
        sample_u = sample_u.reshape(-1) * half_length
        sample_v = sample_v.reshape(-1) * half_width
        best_yaws = []
        best_rolls = []
        for env_index in range(x_index.shape[0]):
            xi = int(x_index[env_index].item())
            zi = int(z_index[env_index].item())
            x_min = max(0, xi - max_dx)
            x_max = min(self.volume_size[0] - 1, xi + max_dx)
            z_min = max(0, zi - max_dz)
            z_max = min(self.volume_size[2] - 1, zi + max_dz)
            local = information[env_index, x_min : x_max + 1, z_min : z_max + 1]
            dx = (torch.arange(x_min, x_max + 1, device=self.device).float() - float(xi)) * cell_size
            dz = (torch.arange(z_min, z_max + 1, device=self.device).float() - float(zi)) * cell_size
            grid_dx, grid_dz = torch.meshgrid(dx, dz, indexing="ij")
            scores = []
            poses = []
            cmd_x_value = float(cmd_x[env_index, xi, zi].item())
            cmd_z_value = float(cmd_z[env_index, xi, zi].item())
            human_index = int(rec.env_to_human_inds[env_index].item())
            label_x = int(torch.clamp(torch.tensor(cmd_x_value, device=self.device), 0, rec.label_maps[human_index].shape[0] - 1).item())
            label_z = int(torch.clamp(torch.tensor(cmd_z_value, device=self.device), 0, rec.label_maps[human_index].shape[2] - 1).item())
            surface_y = float(rec.surface_map_list[human_index][label_x, label_z].item())
            normal = rec.surface_normal_list[human_index][label_x, label_z].float().to(self.device)
            normal = normal / normal.norm().clamp_min(1e-6)
            corner = rec.human_rec_volume_corner[env_index].float()
            center_label = torch.tensor([cmd_x_value, surface_y, cmd_z_value], dtype=torch.float32, device=self.device)
            center_volume = (center_label * float(rec.label_res) - corner) / float(rec.volume_res)
            for yaw in self.yaw + self.yaw_offsets:
                direction_x = torch.cos(yaw)
                direction_z = torch.sin(yaw)
                along = grid_dx * direction_x + grid_dz * direction_z
                across = -grid_dx * direction_z + grid_dz * direction_x
                strip = (along.abs() <= half_length) & (across.abs() <= half_width)
                x_axis_proj = torch.stack([direction_x, torch.zeros_like(direction_x), direction_z])
                y_axis = torch.cross(normal, x_axis_proj, dim=0)
                if y_axis.norm() < 1e-6:
                    x_axis_proj = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
                    y_axis = torch.cross(normal, x_axis_proj, dim=0)
                if y_axis.norm() < 1e-6:
                    x_axis_proj = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
                    y_axis = torch.cross(normal, x_axis_proj, dim=0)
                y_axis = y_axis / y_axis.norm().clamp_min(1e-6)
                x_axis = torch.cross(y_axis, normal, dim=0)
                x_axis = x_axis / x_axis.norm().clamp_min(1e-6)
                for roll in self.roll + self.roll_offsets:
                    rot_y_axis = y_axis * torch.cos(roll) + normal * torch.sin(roll)
                    rot_z_axis = normal * torch.cos(roll) - y_axis * torch.sin(roll)
                    image_center = center_volume + (
                        rot_z_axis
                        * (float(getattr(rec, "height_img", 0.0)) + float(getattr(rec, "add_height", 0.0)) - float(getattr(rec, "height", 0.0)))
                        / float(rec.volume_res)
                    )
                    offsets = sample_u.unsqueeze(1) * x_axis.unsqueeze(0) + sample_v.unsqueeze(1) * rot_y_axis.unsqueeze(0)
                    sample_volume = image_center.unsqueeze(0) + offsets * float(rec.label_res) / float(rec.volume_res)
                    sample_index = sample_volume.round().long()
                    valid = (
                        (sample_index[:, 0] >= 0)
                        & (sample_index[:, 0] < self.volume_size[0])
                        & (sample_index[:, 1] >= 0)
                        & (sample_index[:, 1] < self.volume_size[1])
                        & (sample_index[:, 2] >= 0)
                        & (sample_index[:, 2] < self.volume_size[2])
                    )
                    pose_score = local.new_tensor(0.0)
                    if valid.any():
                        valid_index = sample_index[valid]
                        pose_score = volume_information[env_index, valid_index[:, 0], valid_index[:, 1], valid_index[:, 2]].sum()
                    score = (1.0 - 0.35) * (local * strip.float()).sum() + 0.35 * pose_score
                    score = score - 1.0e-4 * (torch.abs(yaw - self.yaw) + torch.abs(roll - self.roll))
                    scores.append(score)
                    poses.append((yaw, roll))
            best_yaw, best_roll = poses[int(torch.stack(scores).argmax().item())]
            best_yaws.append(best_yaw)
            best_rolls.append(best_roll)
        return torch.stack(best_yaws).float(), torch.stack(best_rolls).float()

    def _update_belief(self, cur_cmd_state):
        import torch

        cmd_x, cmd_z = self._cmd_grid()
        cur_x = cur_cmd_state[:, 0].reshape(-1, 1, 1)
        cur_z = cur_cmd_state[:, 1].reshape(-1, 1, 1)
        dist = torch.sqrt((cmd_x - cur_x).pow(2) + (cmd_z - cur_z).pow(2))
        observed = dist <= self.observation_radius
        self.visit_counts += observed.float()
        covered = self._covered_xz_mask()
        covered_count = covered.sum(dim=(1, 2))
        if self.last_covered_count is not None:
            gained = covered_count > self.last_covered_count
            self.stagnation_steps = torch.where(gained, torch.zeros_like(self.stagnation_steps), self.stagnation_steps + 1)
        self.last_covered_count = covered_count
        target_support = self.prior_xz > self.prior_threshold
        self.uncertainty = torch.where(covered | observed, self.uncertainty * 0.35, self.uncertainty)
        self.uncertainty = torch.where(target_support, self.uncertainty, torch.zeros_like(self.uncertainty))
        frontier = self._frontier_mask(covered, target_support)
        return covered, target_support, frontier

    def goal(self, cur_cmd_state, step: int):
        import torch

        self._maybe_update_patient_prior(step)
        cur_cmd_state = cur_cmd_state.to(self.device).float()
        covered, target_support, frontier = self._update_belief(cur_cmd_state)
        if self.cached_goal is not None and step % self.replan_interval != 0:
            return self.cached_goal
        cmd_x, cmd_z = self._cmd_grid()
        cur_x = cur_cmd_state[:, 0].reshape(-1, 1, 1)
        cur_z = cur_cmd_state[:, 1].reshape(-1, 1, 1)
        dist = torch.sqrt((cmd_x - cur_x).pow(2) + (cmd_z - cur_z).pow(2))
        uncovered = target_support & (~covered)
        information = (
            self.coverage_weight * uncovered.float()
            + self.uncertainty_weight * self.uncertainty
            + self.frontier_weight * frontier.float()
            + self.prior_weight * self.prior_xz
        )
        shaped_score = information * torch.exp(-self.distance_weight * dist)
        shaped_score = shaped_score / (1.0 + self.revisit_weight * self.visit_counts)
        xz_range = torch.tensor(self.task_env.sim_cfg["patient_xz_range"], dtype=torch.float32, device=self.device)
        in_range = (cmd_x >= xz_range[0, 0]) & (cmd_x <= xz_range[1, 0]) & (cmd_z >= xz_range[0, 1]) & (cmd_z <= xz_range[1, 1])
        reachable = dist <= self.reach_radius
        valid = in_range & reachable & (target_support | (self.prior_xz > 0.0))
        shaped_score = torch.where(valid, shaped_score, shaped_score.new_full((), -1.0))
        flat_index = shaped_score.reshape(shaped_score.shape[0], -1).argmax(dim=1)
        x_index = flat_index // shaped_score.shape[2]
        z_index = flat_index % shaped_score.shape[2]
        env_index = torch.arange(shaped_score.shape[0], device=self.device)
        self.visit_counts[env_index, x_index, z_index] += 2.0
        rec = self.task_env.surface_reconstructor
        corner = rec.human_rec_volume_corner
        goal_x = (corner[:, 0] + x_index.float() * float(rec.volume_res)) / float(rec.label_res)
        goal_z = (corner[:, 2] + z_index.float() * float(rec.volume_res)) / float(rec.label_res)
        goal_yaw, goal_roll = self._pose_for_indices(x_index, z_index, information)
        if getattr(rec, "max_roll_adj", None) is not None:
            goal_roll = torch.clamp(goal_roll, min=-rec.max_roll_adj.reshape(-1), max=rec.max_roll_adj.reshape(-1))
        goal = torch.stack(
            [
                goal_x.clamp(float(xz_range[0, 0]), float(xz_range[1, 0])),
                goal_z.clamp(float(xz_range[0, 1]), float(xz_range[1, 1])),
                goal_yaw,
                goal_roll,
            ],
            dim=-1,
        )
        self.cached_goal = goal.detach().cpu()
        return self.cached_goal


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-robot-US-reconstruction-v0")
parser.add_argument("--patient_ids", default=None)
parser.add_argument("--split", choices=("train", "test"), default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--anatomy_prior", default="artifacts/checkpoints/anatomy_prior_l4_train.pt")
parser.add_argument("--output", default="artifacts/trajectories/best_registration_slam.pt")
parser.add_argument("--summary_json", default=None)
parser.add_argument("--auto_install_registration_deps", action="store_true")
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

patients = _resolve_patient_ids(args.patient_ids, args.split)
os.environ["SONOGYM_PATIENT_IDS"] = ",".join(patients)
_ensure_registration_deps(args.auto_install_registration_deps)

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import spinal_surgery  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from spinal_surgery.lab.controllers.heuristic_reconstruction import HeuristicReconstruction


def _patient_ids_for_envs(task_env) -> list[str]:
    rec = task_env.surface_reconstructor
    human_ids = [os.path.basename(path.rstrip("/")) for path in rec.human_list]
    env_to_human = rec.env_to_human_inds.detach().cpu().tolist()
    return [human_ids[int(index)] for index in env_to_human]


def _volume_voxels(task_env):
    return task_env.surface_reconstructor.human_rec_volume.sum(dim=(1, 2, 3))


def _make_action_helper(task_env) -> HeuristicReconstruction:
    human_pos_2d_min = task_env.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]] - 25.0
    human_pos_2d_max = task_env.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]] + 25.0
    return HeuristicReconstruction(
        max_action=task_env.max_action,
        action_scale=task_env.action_scale,
        human_pos_2d_min=human_pos_2d_min,
        human_pos_2d_max=human_pos_2d_max,
        num_sections=2,
        total_steps=args.steps,
        ratio=[0.05, 0.05, 0.05, 0.0],
        device=task_env.sim.device,
    )


def main() -> int:
    prior_path = Path(args.anatomy_prior)
    if not prior_path.exists():
        build_anatomy_prior(prior_path, TRAIN_PATIENTS)
    anatomy_prior = torch.load(prior_path, map_location=args.device)
    prior_volume = anatomy_prior["prior_volume"].float()

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg)
    task_env = env.unwrapped
    registration_prior = RegistrationPriorPredictor(prior_volume.float())
    planner = ActiveUSSLAMGoalPlanner(task_env, prior_volume, prior_predictor=registration_prior)
    action_helper = _make_action_helper(task_env)

    _, info = env.reset()
    planner.reset()
    patient_ids_for_envs = _patient_ids_for_envs(task_env)
    cmd_states = []
    goals = []
    actions_list = []
    reconstructed = []
    delta_reconstructed = []
    coverage_trace = []
    blend_weight_trace = []
    stats_trace = []

    for step in range(args.steps):
        goal = planner.goal(info["cur_cmd_state"], step).to(task_env.sim.device)
        actions = action_helper.get_action_given_goal(info, goal)
        before = _volume_voxels(task_env)
        cmd_states.append(info["cur_cmd_state"].detach().cpu())
        goals.append(goal.detach().cpu())
        actions_list.append(actions.detach().cpu())
        _, _, terminated, truncated, info = env.step(actions)
        after = _volume_voxels(task_env)
        reconstructed.append(after.detach().cpu())
        delta_reconstructed.append((after - before).clamp_min(0.0).detach().cpu())
        coverage_trace.append(task_env.surface_reconstructor.get_converage_ratio().detach().cpu())
        blend_weight_trace.append(registration_prior.blend_weight(task_env.surface_reconstructor.human_rec_volume).detach().cpu())
        stats_trace.append(list(registration_prior.last_registration_stats))
        if torch.any(torch.logical_or(terminated, truncated)).item() and step < args.steps - 1:
            print(f"[WARN] rollout ended at step {step + 1}; requested {args.steps}.")
            break

    final_coverage = task_env.surface_reconstructor.get_converage_ratio().detach().cpu()
    output = {
        "kind": "best_registration_prior_aus_slam_rollout",
        "algorithm": "aus_slam_registration_adaptive_xz_and_surface_constrained_se3",
        "patient_ids": patient_ids_for_envs,
        "steps": len(reconstructed),
        "cmd_state": torch.stack(cmd_states, dim=1),
        "goal_cmd_pose": torch.stack(goals, dim=1),
        "action": torch.stack(actions_list, dim=1),
        "reconstructed_volume_voxels": torch.stack(reconstructed, dim=1),
        "delta_reconstructed_volume_voxels": torch.stack(delta_reconstructed, dim=1),
        "coverage_trace": torch.stack(coverage_trace, dim=1),
        "final_coverage": final_coverage,
        "mean_final_coverage": float(final_coverage.mean().item()),
        "blend_weight_trace": torch.stack(blend_weight_trace, dim=1),
        "registration_stats_trace": stats_trace,
        "anatomy_prior": str(prior_path),
        "anatomy_prior_volume": anatomy_prior["prior_volume"].detach().cpu(),
        "final_registration_prior_volume": planner.prior_volume.detach().cpu(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    summary = {
        "algorithm": output["algorithm"],
        "patient_ids": patient_ids_for_envs,
        "steps": output["steps"],
        "final_coverage": [float(value) for value in final_coverage.tolist()],
        "mean_final_coverage": output["mean_final_coverage"],
        "output": str(output_path),
    }
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[RESULT] saved rollout to {output_path}")
    print(f"[RESULT] saved summary to {summary_path}")
    print(f"[RESULT] mean_final_coverage={output['mean_final_coverage']:.4f}")
    env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()

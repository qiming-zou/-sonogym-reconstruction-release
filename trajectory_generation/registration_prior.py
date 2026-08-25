"""No-training partial-to-template registration prior for AUS-SLAM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


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


def _nearest_confidence(
    transformed_template: np.ndarray,
    observed: np.ndarray,
    inlier_radius: float,
) -> tuple[float, float]:
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


def _open3d_coarse_transform(
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float,
) -> np.ndarray | None:
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
        radius_normal = voxel_size * 2.0
        radius_feature = voxel_size * 5.0
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
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
    if not np.isfinite(transform).all():
        return None
    return transform


def _probreg_refine(
    source: np.ndarray,
    target: np.ndarray,
    maxiter: int,
) -> np.ndarray:
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
    """Predict a patient prior by registering an atlas/template to observed points."""

    global_prior: torch.Tensor
    min_observed_points: int = 30
    max_template_points: int = 1200
    max_observed_points: int = 900
    prior_threshold: float = 0.05
    coarse_voxel_size: float = 3.0
    inlier_radius: float = 3.0
    max_blend_weight: float = 0.55
    cpd_maxiter: int = 25

    def __post_init__(self) -> None:
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
        self.last_blend_weights: torch.Tensor | None = None
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
        stats = {
            "observed_points": int(observed.shape[0]),
            "fitness": float(fitness),
            "rmse": float(rmse),
            "blend_weight": float(blend_weight),
        }

        full_coarse = self.template_points @ coarse[:3, :3].T + coarse[:3, 3]
        if refined_source.shape[0] == coarse_source.shape[0] and coarse_source.shape[0] > 0:
            displacement = refined_source - coarse_source
            shift = displacement.mean(axis=0)
            full_registered = full_coarse + shift
        else:
            full_registered = full_coarse
        return full_registered, float(blend_weight), stats

    def _rasterize(self, points: np.ndarray, weights: np.ndarray, shape: tuple[int, int, int]) -> torch.Tensor:
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

    @torch.inference_mode()
    def predict(self, sparse_reconstruction: torch.Tensor, patient_ids: list[str] | None = None) -> torch.Tensor:
        sparse = sparse_reconstruction.detach().float().cpu()
        batch_size = sparse.shape[0]
        shape = tuple(int(v) for v in sparse.shape[-3:])
        priors = []
        blend_weights = []
        stats = []
        for env_index in range(batch_size):
            observed = (sparse[env_index] > 0).nonzero(as_tuple=False).float().numpy().astype(np.float64)
            if observed.shape[0] < self.min_observed_points:
                priors.append(self.global_prior.clone())
                blend_weights.append(0.0)
                stats.append(
                    {
                        "observed_points": int(observed.shape[0]),
                        "fitness": 0.0,
                        "rmse": float("inf"),
                        "blend_weight": 0.0,
                    }
                )
                continue
            registered, blend_weight, env_stats = self._register(observed)
            priors.append(self._rasterize(registered, self.template_weights, shape))
            blend_weights.append(blend_weight)
            stats.append(env_stats)
        self.last_blend_weights = torch.tensor(blend_weights, dtype=torch.float32)
        self.last_registration_stats = stats
        return torch.stack(priors, dim=0).to(sparse_reconstruction.device)

    def blend_weight(self, sparse_reconstruction: torch.Tensor, patient_ids: list[str] | None = None) -> torch.Tensor:
        if self.last_blend_weights is None or self.last_blend_weights.shape[0] != sparse_reconstruction.shape[0]:
            return torch.zeros((sparse_reconstruction.shape[0],), dtype=torch.float32)
        return self.last_blend_weights.to(sparse_reconstruction.device)

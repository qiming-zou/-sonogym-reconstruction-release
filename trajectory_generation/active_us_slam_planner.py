"""Active ultrasound SLAM-style next-best-view planner.

The planner treats robotic ultrasound reconstruction as active mapping.  It
maintains a lightweight belief over target x-z cells:

- covered map: derived online from ``surface_reconstructor.human_rec_volume``.
- uncertainty map: high for target-prior cells that have not been observed.
- visited map: penalizes repeated scanning near previously selected goals.
- frontier map: unobserved target-prior cells adjacent to covered cells.

It returns command-space goals for the existing local controller.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


class ActiveUSSLAMGoalPlanner:
    """Frontier and uncertainty driven active reconstruction planner."""

    def __init__(
        self,
        task_env,
        prior_volume: torch.Tensor | None,
        replan_interval: int,
        reach_radius: float,
        observation_radius: float,
        coverage_weight: float,
        uncertainty_weight: float,
        frontier_weight: float,
        prior_weight: float,
        distance_weight: float,
        revisit_weight: float,
        yaw: float,
        yaw_candidates: int = 1,
        yaw_span: float = 0.0,
        yaw_gain_weight: float = 0.35,
        yaw_strip_length: float = 35.0,
        yaw_strip_width: float = 10.0,
        roll: float = 0.0,
        roll_candidates: int = 1,
        roll_span: float = 0.0,
        pose_score_samples: int = 9,
        prior_predictor=None,
        prior_update_interval: int = 0,
        view_gain_predictor=None,
    ):
        self.task_env = task_env
        self.replan_interval = max(1, int(replan_interval))
        self.reach_radius = float(reach_radius)
        self.observation_radius = float(observation_radius)
        self.coverage_weight = float(coverage_weight)
        self.uncertainty_weight = float(uncertainty_weight)
        self.frontier_weight = float(frontier_weight)
        self.prior_weight = float(prior_weight)
        self.distance_weight = float(distance_weight)
        self.revisit_weight = float(revisit_weight)
        self.yaw = float(yaw)
        self.yaw_candidates = max(1, int(yaw_candidates))
        self.yaw_span = max(0.0, float(yaw_span))
        self.yaw_gain_weight = float(yaw_gain_weight)
        self.yaw_strip_length = float(yaw_strip_length)
        self.yaw_strip_width = float(yaw_strip_width)
        self.roll = float(roll)
        self.roll_candidates = max(1, int(roll_candidates))
        self.roll_span = max(0.0, float(roll_span))
        self.pose_score_samples = max(3, int(pose_score_samples))
        self.prior_predictor = prior_predictor
        self.prior_update_interval = max(0, int(prior_update_interval))
        self.view_gain_predictor = view_gain_predictor
        self.prior_threshold = 0.05
        self.cached_goal: torch.Tensor | None = None

        rec = task_env.surface_reconstructor
        self.device = task_env.sim.device
        volume_size = tuple(int(value) for value in rec.volume_size.detach().cpu().tolist())
        self.volume_size = volume_size
        x_idx = torch.arange(volume_size[0], device=self.device)
        z_idx = torch.arange(volume_size[2], device=self.device)
        self.x_grid, self.z_grid = torch.meshgrid(x_idx, z_idx, indexing="ij")
        volume_axes = [torch.arange(size, dtype=torch.float32, device=self.device) for size in volume_size]
        vx, vy, vz = torch.meshgrid(*volume_axes, indexing="ij")
        self.volume_coords = torch.stack([vx, vy, vz], dim=-1)
        self.visit_counts = torch.zeros((task_env.scene.num_envs, volume_size[0], volume_size[2]), device=self.device)
        self.uncertainty = torch.ones_like(self.visit_counts)
        self.stagnation_steps = torch.zeros((task_env.scene.num_envs,), dtype=torch.long, device=self.device)
        self.last_covered_count: torch.Tensor | None = None

        if prior_volume is not None:
            self.prior_volume = self._normalize_prior_volume(prior_volume.float().to(self.device), task_env.scene.num_envs)
            self.prior_xz = self._project_prior_volume(self.prior_volume, task_env.scene.num_envs)
        else:
            self.prior_volume = self._target_volume().float()
            self.prior_xz = self._target_xz_mask().float()
        self.base_prior_xz = self.prior_xz.clone()
        self.base_prior_volume = self.prior_volume.clone()
        self.patient_prior_stagnation_steps = 50

        self.kernel = torch.ones((1, 1, 3, 3), device=self.device)
        self.yaw_offsets = self._make_yaw_offsets()
        self.roll_offsets = self._make_roll_offsets()

    def reset(self) -> None:
        self.cached_goal = None
        self.visit_counts.zero_()
        self.uncertainty.fill_(1.0)
        self.stagnation_steps.zero_()
        self.last_covered_count = None
        self.prior_xz = self.base_prior_xz.clone()
        self.prior_volume = self.base_prior_volume.clone()

    def _make_yaw_offsets(self) -> torch.Tensor:
        if self.yaw_candidates <= 1 or self.yaw_span <= 0.0:
            return torch.zeros((1,), dtype=torch.float32, device=self.device)
        return torch.linspace(
            -0.5 * self.yaw_span,
            0.5 * self.yaw_span,
            steps=self.yaw_candidates,
            dtype=torch.float32,
            device=self.device,
        )

    def _make_roll_offsets(self) -> torch.Tensor:
        if self.roll_candidates <= 1 or self.roll_span <= 0.0:
            return torch.zeros((1,), dtype=torch.float32, device=self.device)
        return torch.linspace(
            -0.5 * self.roll_span,
            0.5 * self.roll_span,
            steps=self.roll_candidates,
            dtype=torch.float32,
            device=self.device,
        )

    def _uses_full_pose_goal(self) -> bool:
        return self.roll_offsets.numel() > 1 or self.roll_span > 0.0

    def _normalize_prior_volume(self, prior_volume: torch.Tensor, num_envs: int) -> torch.Tensor:
        if prior_volume.ndim == 3:
            volume = prior_volume.unsqueeze(0).repeat(num_envs, 1, 1, 1)
        elif prior_volume.ndim == 4:
            if prior_volume.shape[0] == num_envs:
                volume = prior_volume
            elif prior_volume.shape[0] == 1:
                volume = prior_volume.repeat(num_envs, 1, 1, 1)
            else:
                raise ValueError(
                    f"Expected prior batch size 1 or {num_envs}, got {prior_volume.shape[0]}."
                )
        else:
            raise ValueError(f"Expected prior volume shape (X,Y,Z) or (B,X,Y,Z), got {tuple(prior_volume.shape)}")
        return volume.float() / volume.float().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)

    def _project_prior_volume(self, prior_volume: torch.Tensor, num_envs: int) -> torch.Tensor:
        if prior_volume.ndim == 3:
            prior_xz = prior_volume.float().sum(dim=1)
            prior_xz = prior_xz / prior_xz.amax().clamp_min(1e-6)
            return prior_xz.unsqueeze(0).repeat(num_envs, 1, 1)
        if prior_volume.ndim == 4:
            prior_xz = prior_volume.float().sum(dim=2)
            prior_xz = prior_xz / prior_xz.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            return prior_xz
        raise ValueError(f"Expected prior volume shape (X,Y,Z) or (B,X,Y,Z), got {tuple(prior_volume.shape)}")

    def _maybe_update_patient_prior(self, step: int) -> None:
        if self.prior_predictor is None:
            return
        if self.prior_update_interval <= 0 and step > 0:
            return
        if self.prior_update_interval > 0 and step % self.prior_update_interval != 0:
            return
        rec_volume = self.task_env.surface_reconstructor.human_rec_volume.detach().float()
        predicted_prior = self.prior_predictor.predict(rec_volume, self._patient_ids_for_envs()).to(self.device)
        predicted_volume = self._normalize_prior_volume(predicted_prior, rec_volume.shape[0])
        predicted_xz = self._project_prior_volume(predicted_volume, rec_volume.shape[0])
        if hasattr(self.prior_predictor, "blend_weight"):
            patient_weight = self.prior_predictor.blend_weight(rec_volume, self._patient_ids_for_envs()).to(self.device)
        else:
            patient_weight = torch.where(
                self.stagnation_steps >= self.patient_prior_stagnation_steps,
                torch.full_like(self.stagnation_steps, 0.35, dtype=torch.float32),
                torch.zeros_like(self.stagnation_steps, dtype=torch.float32),
            )
        patient_weight = patient_weight.float().clamp(0.0, 1.0)
        patient_weight = patient_weight.reshape(-1, 1, 1)
        blended_prior = (1.0 - patient_weight) * self.base_prior_xz + patient_weight * predicted_xz
        self.prior_xz = blended_prior / blended_prior.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        patient_weight_3d = patient_weight.reshape(-1, 1, 1, 1)
        blended_volume = (1.0 - patient_weight_3d) * self.base_prior_volume + patient_weight_3d * predicted_volume
        self.prior_volume = blended_volume / blended_volume.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)

    def _patient_ids_for_envs(self) -> list[str]:
        rec = self.task_env.surface_reconstructor
        human_ids = [os.path.basename(path.rstrip("/")) for path in rec.human_list]
        env_to_human = rec.env_to_human_inds.detach().cpu().tolist()
        return [human_ids[int(index)] for index in env_to_human]

    def _target_xz_mask(self) -> torch.Tensor:
        return self._target_volume().amax(dim=2) > 0

    def _target_volume(self) -> torch.Tensor:
        rec = self.task_env.surface_reconstructor
        target_stack = torch.stack(rec.upper_surface_volume_list, dim=0).to(self.device)
        env_to_human = rec.env_to_human_inds.long().to(self.device)
        return target_stack[env_to_human]

    def _covered_xz_mask(self) -> torch.Tensor:
        rec = self.task_env.surface_reconstructor
        return rec.human_rec_volume.float().amax(dim=2) > 0

    def _frontier_mask(self, covered: torch.Tensor, target_support: torch.Tensor) -> torch.Tensor:
        covered_f = covered.float().unsqueeze(1)
        neighbor_count = F.conv2d(covered_f, self.kernel, padding=1).squeeze(1)
        return target_support & (~covered) & (neighbor_count > 0)

    def _cmd_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.task_env.surface_reconstructor
        corner = rec.human_rec_volume_corner
        cmd_x = (corner[:, 0:1, None] + self.x_grid.unsqueeze(0) * float(rec.volume_res)) / float(rec.label_res)
        cmd_z = (corner[:, 2:3, None] + self.z_grid.unsqueeze(0) * float(rec.volume_res)) / float(rec.label_res)
        return cmd_x, cmd_z

    def _pose_for_indices(
        self,
        x_index: torch.Tensor,
        z_index: torch.Tensor,
        information: torch.Tensor,
        cur_cmd_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.yaw_offsets.numel() <= 1
            and self.roll_offsets.numel() <= 1
        ) or self.yaw_gain_weight <= 0.0:
            base_yaw = torch.full_like(x_index.float(), self.yaw)
            base_roll = torch.full_like(x_index.float(), self.roll)
            return base_yaw, base_roll

        half_length = 0.5 * self.yaw_strip_length
        half_width = 0.5 * self.yaw_strip_width
        rec = self.task_env.surface_reconstructor
        cell_size = max(float(rec.volume_res) / max(float(rec.label_res), 1e-6), 1e-6)
        max_dx = max(1, int(round(half_length / cell_size)))
        max_dz = max(1, int(round(half_length / cell_size)))
        best_yaws = []
        best_rolls = []
        cmd_x, cmd_z = self._cmd_grid()
        volume_information = torch.where(
            self.task_env.surface_reconstructor.human_rec_volume.float() > 0,
            torch.zeros_like(self.prior_volume),
            self.prior_volume,
        )
        sample_axis = torch.linspace(
            -1.0,
            1.0,
            steps=self.pose_score_samples,
            dtype=torch.float32,
            device=self.device,
        )
        sample_u, sample_v = torch.meshgrid(sample_axis, sample_axis, indexing="ij")
        sample_u = sample_u.reshape(-1) * half_length
        sample_v = sample_v.reshape(-1) * half_width
        patient_ids_for_envs = self._patient_ids_for_envs() if self.view_gain_predictor is not None else None

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
            strip_scores = []
            plane_masks = []
            candidate_penalties = []
            candidate_poses = []
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
                    offsets = (
                        sample_u.unsqueeze(1) * x_axis.unsqueeze(0)
                        + sample_v.unsqueeze(1) * rot_y_axis.unsqueeze(0)
                    )
                    sample_volume = image_center.unsqueeze(0) + offsets * float(rec.label_res) / float(rec.volume_res)
                    if self.view_gain_predictor is not None:
                        label_to_volume = float(rec.label_res) / float(rec.volume_res)
                        rel_volume = self.volume_coords - image_center.reshape(1, 1, 1, 3)
                        dense_along = (rel_volume * x_axis.reshape(1, 1, 1, 3)).sum(dim=-1)
                        dense_across = (rel_volume * rot_y_axis.reshape(1, 1, 1, 3)).sum(dim=-1)
                        dense_distance = (rel_volume * rot_z_axis.reshape(1, 1, 1, 3)).sum(dim=-1).abs()
                        plane_masks.append(
                            (
                                (dense_along.abs() <= half_length * label_to_volume)
                                & (dense_across.abs() <= half_width * label_to_volume)
                                & (dense_distance <= 0.75)
                            ).float()
                        )
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
                        pose_score = volume_information[
                            env_index,
                            valid_index[:, 0],
                            valid_index[:, 1],
                            valid_index[:, 2],
                        ].sum()
                    score = (1.0 - self.yaw_gain_weight) * (local * strip.float()).sum() + self.yaw_gain_weight * pose_score
                    penalty = 1.0e-4 * (torch.abs(yaw - self.yaw) + torch.abs(roll - self.roll))
                    score = score - penalty
                    scores.append(score)
                    strip_scores.append((local * strip.float()).sum())
                    candidate_penalties.append(penalty)
                    candidate_poses.append((yaw, roll))
            if self.view_gain_predictor is not None and plane_masks:
                plane_batch = torch.stack(plane_masks, dim=0).unsqueeze(0)
                rec_volume = rec.human_rec_volume[env_index : env_index + 1].detach().float()
                prior_volume = self.prior_volume[env_index : env_index + 1].detach().float()
                gain_scores = self.view_gain_predictor.predict(
                    rec_volume,
                    plane_batch,
                    [patient_ids_for_envs[env_index]] if patient_ids_for_envs is not None else None,
                    prior_volume,
                ).reshape(-1).to(self.device)
                strip_tensor = torch.stack(strip_scores).float()
                strip_tensor = strip_tensor / strip_tensor.amax().clamp_min(1e-6)
                penalties = torch.stack(candidate_penalties).float()
                score_tensor = gain_scores + 0.05 * strip_tensor - penalties
            else:
                score_tensor = torch.stack(scores)
            best_index = int(score_tensor.argmax().item())
            best_yaw, best_roll = candidate_poses[best_index]
            best_yaws.append(best_yaw)
            best_rolls.append(best_roll)
        return torch.stack(best_yaws).float(), torch.stack(best_rolls).float()

    def _update_belief(self, cur_cmd_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def goal(self, cur_cmd_state: torch.Tensor, step: int) -> torch.Tensor:
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

        xz_range = torch.tensor(
            self.task_env.sim_cfg["patient_xz_range"],
            dtype=torch.float32,
            device=self.device,
        )
        in_range = (
            (cmd_x >= xz_range[0, 0])
            & (cmd_x <= xz_range[1, 0])
            & (cmd_z >= xz_range[0, 1])
            & (cmd_z <= xz_range[1, 1])
        )
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
        goal_yaw, goal_roll = self._pose_for_indices(x_index, z_index, information, cur_cmd_state)
        goal_components = [
            goal_x.clamp(float(xz_range[0, 0]), float(xz_range[1, 0])),
            goal_z.clamp(float(xz_range[0, 1]), float(xz_range[1, 1])),
            goal_yaw,
        ]
        if self._uses_full_pose_goal():
            max_roll = getattr(rec, "max_roll_adj", None)
            if max_roll is not None:
                goal_roll = torch.clamp(goal_roll, min=-max_roll.reshape(-1), max=max_roll.reshape(-1))
            goal_components.append(goal_roll)
        goal = torch.stack(goal_components, dim=-1)
        self.cached_goal = goal.detach().cpu()
        return self.cached_goal


class RecedingHorizonActiveUSSLAMGoalPlanner(ActiveUSSLAMGoalPlanner):
    """Short-horizon informative path planner for active US reconstruction.

    Unlike ``ActiveUSSLAMGoalPlanner``, which greedily selects one best target
    cell, this planner searches over short waypoint sequences and executes the
    first waypoint of the best sequence.  The search objective rewards expected
    coverage and uncertainty reduction along a continuous scan while penalizing
    travel and repeated visits.
    """

    def __init__(
        self,
        task_env,
        prior_volume: torch.Tensor | None,
        replan_interval: int,
        reach_radius: float,
        observation_radius: float,
        coverage_weight: float,
        uncertainty_weight: float,
        frontier_weight: float,
        prior_weight: float,
        distance_weight: float,
        revisit_weight: float,
        yaw: float,
        horizon: int,
        branch_factor: int,
        beam_width: int,
        step_radius: float,
        path_length_weight: float,
        overlap_weight: float,
        yaw_candidates: int = 1,
        yaw_span: float = 0.0,
        yaw_gain_weight: float = 0.35,
        yaw_strip_length: float = 35.0,
        yaw_strip_width: float = 10.0,
        roll: float = 0.0,
        roll_candidates: int = 1,
        roll_span: float = 0.0,
        pose_score_samples: int = 9,
        prior_predictor=None,
        prior_update_interval: int = 0,
        view_gain_predictor=None,
    ):
        super().__init__(
            task_env,
            prior_volume,
            replan_interval,
            reach_radius,
            observation_radius,
            coverage_weight,
            uncertainty_weight,
            frontier_weight,
            prior_weight,
            distance_weight,
            revisit_weight,
            yaw,
            yaw_candidates,
            yaw_span,
            yaw_gain_weight,
            yaw_strip_length,
            yaw_strip_width,
            roll,
            roll_candidates,
            roll_span,
            pose_score_samples,
            prior_predictor,
            prior_update_interval,
            view_gain_predictor,
        )
        self.horizon = max(1, int(horizon))
        self.branch_factor = max(1, int(branch_factor))
        self.beam_width = max(1, int(beam_width))
        self.step_radius = float(step_radius)
        self.path_length_weight = float(path_length_weight)
        self.overlap_weight = float(overlap_weight)
        self.observation_kernel = self._make_observation_kernel()

    def _make_observation_kernel(self) -> torch.Tensor:
        rec = self.task_env.surface_reconstructor
        cell_size = float(rec.volume_res) / float(rec.label_res)
        radius_cells = max(1, int(round(self.observation_radius / max(cell_size, 1e-6))))
        axis = torch.arange(-radius_cells, radius_cells + 1, device=self.device)
        dx, dz = torch.meshgrid(axis, axis, indexing="ij")
        disk = ((dx.float().pow(2) + dz.float().pow(2)).sqrt() <= radius_cells).float()
        return disk.reshape(1, 1, disk.shape[0], disk.shape[1])

    def _information_map(
        self,
        covered: torch.Tensor,
        target_support: torch.Tensor,
        frontier: torch.Tensor,
    ) -> torch.Tensor:
        uncovered = target_support & (~covered)
        information = (
            self.coverage_weight * uncovered.float()
            + self.uncertainty_weight * self.uncertainty
            + self.frontier_weight * frontier.float()
            + self.prior_weight * self.prior_xz
        )
        return torch.where(target_support | (self.prior_xz > 0.0), information, torch.zeros_like(information))

    def _valid_map(
        self,
        cmd_x: torch.Tensor,
        cmd_z: torch.Tensor,
        target_support: torch.Tensor,
    ) -> torch.Tensor:
        xz_range = torch.tensor(
            self.task_env.sim_cfg["patient_xz_range"],
            dtype=torch.float32,
            device=self.device,
        )
        in_range = (
            (cmd_x >= xz_range[0, 0])
            & (cmd_x <= xz_range[1, 0])
            & (cmd_z >= xz_range[0, 1])
            & (cmd_z <= xz_range[1, 1])
        )
        return in_range & (target_support | (self.prior_xz > 0.0))

    def _local_gain_map(self, information: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        gain = F.conv2d(
            (information * valid.float()).unsqueeze(1),
            self.observation_kernel,
            padding=self.observation_kernel.shape[-1] // 2,
        ).squeeze(1)
        gain = gain / self.observation_kernel.sum().clamp_min(1.0)
        return torch.where(valid, gain, gain.new_full((), -1.0))

    def _best_path_for_env(
        self,
        env_index: int,
        cur_x: float,
        cur_z: float,
        cmd_x: torch.Tensor,
        cmd_z: torch.Tensor,
        local_gain: torch.Tensor,
        information: torch.Tensor,
    ) -> tuple[int, int]:
        grid_shape = local_gain.shape
        neg_inf = local_gain.new_full((), -1.0e9)
        beams: list[tuple[float, int | None, int | None, tuple[tuple[int, int], ...], float, float]] = [
            (0.0, None, None, tuple(), cur_x, cur_z)
        ]

        for depth in range(self.horizon):
            candidates: list[tuple[float, int, int, tuple[tuple[int, int], ...], float, float]] = []
            max_step = self.reach_radius if depth == 0 else self.step_radius
            for score, prev_x_idx, prev_z_idx, path, anchor_x, anchor_z in beams:
                dist = torch.sqrt((cmd_x - anchor_x).pow(2) + (cmd_z - anchor_z).pow(2))
                candidate_score = local_gain - self.path_length_weight * dist
                candidate_score = candidate_score - self.distance_weight * dist * (0.35 if depth == 0 else 0.15)
                candidate_score = torch.where(dist <= max_step, candidate_score, neg_inf)

                for path_x, path_z in path:
                    overlap_dist = torch.sqrt((cmd_x - cmd_x[path_x, path_z]).pow(2) + (cmd_z - cmd_z[path_x, path_z]).pow(2))
                    overlap_penalty = self.overlap_weight * torch.exp(
                        -overlap_dist / max(self.observation_radius, 1e-6)
                    )
                    candidate_score = candidate_score - overlap_penalty

                if prev_x_idx is not None and prev_z_idx is not None:
                    previous_dist = torch.sqrt(
                        (cmd_x - cmd_x[prev_x_idx, prev_z_idx]).pow(2)
                        + (cmd_z - cmd_z[prev_x_idx, prev_z_idx]).pow(2)
                    )
                    candidate_score = candidate_score - 0.25 * self.overlap_weight * torch.exp(
                        -previous_dist / max(self.observation_radius, 1e-6)
                    )

                flat = candidate_score.reshape(-1)
                k = min(self.branch_factor, int((flat > neg_inf / 2).sum().item()))
                if k <= 0:
                    continue
                top_values, top_indices = flat.topk(k)
                for value, flat_index in zip(top_values.tolist(), top_indices.tolist()):
                    x_index = int(flat_index // grid_shape[1])
                    z_index = int(flat_index % grid_shape[1])
                    new_path = path + ((x_index, z_index),)
                    terminal_bonus = 0.15 * float(information[x_index, z_index].item()) if depth == self.horizon - 1 else 0.0
                    candidates.append(
                        (
                            score + float(value) + terminal_bonus,
                            x_index,
                            z_index,
                            new_path,
                            float(cmd_x[x_index, z_index].item()),
                            float(cmd_z[x_index, z_index].item()),
                        )
                    )

            if not candidates:
                break
            candidates.sort(key=lambda item: item[0], reverse=True)
            beams = candidates[: self.beam_width]

        best_path = beams[0][3]
        if best_path:
            return best_path[-1]

        flat_index = local_gain.reshape(-1).argmax().item()
        return int(flat_index // grid_shape[1]), int(flat_index % grid_shape[1])

    def goal(self, cur_cmd_state: torch.Tensor, step: int) -> torch.Tensor:
        self._maybe_update_patient_prior(step)
        cur_cmd_state = cur_cmd_state.to(self.device).float()
        covered, target_support, frontier = self._update_belief(cur_cmd_state)

        if self.cached_goal is not None and step % self.replan_interval != 0:
            return self.cached_goal

        cmd_x, cmd_z = self._cmd_grid()
        information = self._information_map(covered, target_support, frontier)
        valid = self._valid_map(cmd_x, cmd_z, target_support)
        local_gain = self._local_gain_map(information, valid)
        local_gain = local_gain / (1.0 + self.revisit_weight * self.visit_counts)

        goal_indices = []
        for env_index in range(cur_cmd_state.shape[0]):
            x_index, z_index = self._best_path_for_env(
                env_index,
                float(cur_cmd_state[env_index, 0].item()),
                float(cur_cmd_state[env_index, 1].item()),
                cmd_x[env_index],
                cmd_z[env_index],
                local_gain[env_index],
                information[env_index],
            )
            goal_indices.append((x_index, z_index))

        x_index = torch.tensor([item[0] for item in goal_indices], dtype=torch.long, device=self.device)
        z_index = torch.tensor([item[1] for item in goal_indices], dtype=torch.long, device=self.device)
        env_index = torch.arange(cur_cmd_state.shape[0], device=self.device)
        self.visit_counts[env_index, x_index, z_index] += 2.0

        rec = self.task_env.surface_reconstructor
        corner = rec.human_rec_volume_corner
        xz_range = torch.tensor(
            self.task_env.sim_cfg["patient_xz_range"],
            dtype=torch.float32,
            device=self.device,
        )
        goal_x = (corner[:, 0] + x_index.float() * float(rec.volume_res)) / float(rec.label_res)
        goal_z = (corner[:, 2] + z_index.float() * float(rec.volume_res)) / float(rec.label_res)
        goal_yaw, goal_roll = self._pose_for_indices(x_index, z_index, information, cur_cmd_state)
        goal_components = [
            goal_x.clamp(float(xz_range[0, 0]), float(xz_range[1, 0])),
            goal_z.clamp(float(xz_range[0, 1]), float(xz_range[1, 1])),
            goal_yaw,
        ]
        if self._uses_full_pose_goal():
            max_roll = getattr(rec, "max_roll_adj", None)
            if max_roll is not None:
                goal_roll = torch.clamp(goal_roll, min=-max_roll.reshape(-1), max=max_roll.reshape(-1))
            goal_components.append(goal_roll)
        goal = torch.stack(goal_components, dim=-1)
        self.cached_goal = goal.detach().cpu()
        return self.cached_goal

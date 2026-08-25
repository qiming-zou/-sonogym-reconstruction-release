"""Prior-based online next-best-view goal planner."""

from __future__ import annotations

import torch


class OnlineNBVGoalPlanner:
    """Select reachable command-space goals from anatomy-prior mass not yet visited."""

    def __init__(
        self,
        task_env,
        prior_volume: torch.Tensor,
        replan_interval: int,
        reach_radius: float,
        observation_radius: float,
        distance_weight: float,
        visit_weight: float,
        yaw: float,
    ):
        self.task_env = task_env
        self.prior_volume = prior_volume.float().to(task_env.sim.device)
        self.replan_interval = max(1, int(replan_interval))
        self.reach_radius = float(reach_radius)
        self.observation_radius = float(observation_radius)
        self.distance_weight = float(distance_weight)
        self.visit_weight = float(visit_weight)
        self.yaw = float(yaw)
        self.cached_goal: torch.Tensor | None = None
        volume_size = tuple(int(value) for value in task_env.surface_reconstructor.volume_size.detach().cpu().tolist())
        self.visit_counts = torch.zeros(
            (task_env.scene.num_envs, volume_size[0], volume_size[2]),
            device=task_env.sim.device,
        )
        x_idx = torch.arange(volume_size[0], device=task_env.sim.device)
        z_idx = torch.arange(volume_size[2], device=task_env.sim.device)
        self.x_grid, self.z_grid = torch.meshgrid(x_idx, z_idx, indexing="ij")

    def reset(self) -> None:
        self.cached_goal = None
        self.visit_counts.zero_()

    def goal(self, cur_cmd_state: torch.Tensor, step: int) -> torch.Tensor:
        if self.cached_goal is not None and step % self.replan_interval != 0:
            return self.cached_goal

        rec = self.task_env.surface_reconstructor
        corner = rec.human_rec_volume_corner
        cmd_x = (corner[:, 0:1, None] + self.x_grid.unsqueeze(0) * float(rec.volume_res)) / float(rec.label_res)
        cmd_z = (corner[:, 2:3, None] + self.z_grid.unsqueeze(0) * float(rec.volume_res)) / float(rec.label_res)
        cur_x = cur_cmd_state[:, 0].reshape(-1, 1, 1)
        cur_z = cur_cmd_state[:, 1].reshape(-1, 1, 1)
        dist = torch.sqrt((cmd_x - cur_x).pow(2) + (cmd_z - cur_z).pow(2))
        self.visit_counts += (dist <= self.observation_radius).float()

        score_xz = self.prior_volume.unsqueeze(0).sum(dim=2).repeat(cur_cmd_state.shape[0], 1, 1)

        xz_range = torch.tensor(
            self.task_env.sim_cfg["patient_xz_range"],
            dtype=torch.float32,
            device=self.task_env.sim.device,
        )
        in_range = (
            (cmd_x >= xz_range[0, 0])
            & (cmd_x <= xz_range[1, 0])
            & (cmd_z >= xz_range[0, 1])
            & (cmd_z <= xz_range[1, 1])
        )
        reachable = dist <= self.reach_radius
        shaped_score = score_xz * torch.exp(-self.distance_weight * dist)
        shaped_score = shaped_score / (1.0 + self.visit_weight * self.visit_counts)
        shaped_score = torch.where(in_range & reachable, shaped_score, shaped_score.new_full((), -1.0))

        flat_index = shaped_score.reshape(shaped_score.shape[0], -1).argmax(dim=1)
        x_index = flat_index // shaped_score.shape[2]
        z_index = flat_index % shaped_score.shape[2]

        env_index = torch.arange(shaped_score.shape[0], device=self.task_env.sim.device)
        self.visit_counts[env_index, x_index, z_index] += 1.0

        goal_x = (corner[:, 0] + x_index.float() * float(rec.volume_res)) / float(rec.label_res)
        goal_z = (corner[:, 2] + z_index.float() * float(rec.volume_res)) / float(rec.label_res)
        goal = torch.stack(
            [
                goal_x.clamp(float(xz_range[0, 0]), float(xz_range[1, 0])),
                goal_z.clamp(float(xz_range[0, 1]), float(xz_range[1, 1])),
                torch.full_like(goal_x, self.yaw),
            ],
            dim=-1,
        )
        self.cached_goal = goal.detach().cpu()
        return self.cached_goal

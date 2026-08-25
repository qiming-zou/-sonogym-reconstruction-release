"""Generate reconstruction training trajectories around the target organ."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajectory_generation.patient_splits import DEFAULT_SPLIT_FILE, apply_patient_ids_env, resolve_patient_ids

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate reconstruction trajectories near the target organ.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--task", type=str, default="Isaac-robot-US-reconstruction-v0")
parser.add_argument("--num_traj", type=int, default=64)
parser.add_argument(
    "--trajectory_length",
    type=int,
    default=500,
    help="Number of steps per trajectory.",
)
parser.add_argument("--square_size", type=float, default=25.0)
parser.add_argument("--output", type=str, default="artifacts/trajectories/random_reconstruction_trajectories.pt")
parser.add_argument(
    "--no_save_us_images",
    action="store_true",
    help="Do not save per-step 2D ultrasound images.",
)
parser.add_argument(
    "--us_sim_mode",
    choices=("conv", "net", "both"),
    default="net",
    help="SonoGym ultrasound simulation mode used for saved 2D US images.",
)
parser.add_argument(
    "--anatomy_prior",
    type=str,
    default=None,
    help="Optional anatomy prior checkpoint used to save prior slices and prior-gain rewards.",
)
parser.add_argument(
    "--patient_prior_model",
    type=str,
    default=None,
    help="Optional patient-conditioned prior checkpoint used for online AUS-SLAM belief updates.",
)
parser.add_argument(
    "--view_gain_model",
    type=str,
    default=None,
    help="Optional view-conditioned gain checkpoint used to score AUS-SLAM yaw/roll candidates.",
)
parser.add_argument(
    "--registration_prior",
    action="store_true",
    help="Use no-training partial-to-template registration as the online AUS-SLAM prior.",
)
parser.add_argument("--patient_prior_update_interval", type=int, default=20)
parser.add_argument("--goal_source", choices=("random", "nbv", "expert", "aus_slam", "aus_rh_slam"), default="random")
parser.add_argument("--min_final_coverage", type=float, default=0.0)
parser.add_argument("--max_attempt_batches", type=int, default=32)
parser.add_argument("--random_replan_interval", type=int, default=20)
parser.add_argument("--random_yaw_min", type=float, default=0.0)
parser.add_argument("--random_yaw_max", type=float, default=1.57)
parser.add_argument("--nbv_replan_interval", type=int, default=20)
parser.add_argument("--nbv_reach_radius", type=float, default=70.0)
parser.add_argument("--nbv_observation_radius", type=float, default=25.0)
parser.add_argument("--nbv_distance_weight", type=float, default=0.02)
parser.add_argument("--nbv_visit_weight", type=float, default=0.35)
parser.add_argument("--nbv_yaw", type=float, default=1.57)
parser.add_argument("--aus_replan_interval", type=int, default=10)
parser.add_argument("--aus_reach_radius", type=float, default=75.0)
parser.add_argument("--aus_observation_radius", type=float, default=22.0)
parser.add_argument("--aus_coverage_weight", type=float, default=1.0)
parser.add_argument("--aus_uncertainty_weight", type=float, default=1.5)
parser.add_argument("--aus_frontier_weight", type=float, default=2.0)
parser.add_argument("--aus_prior_weight", type=float, default=0.6)
parser.add_argument("--aus_distance_weight", type=float, default=0.018)
parser.add_argument("--aus_revisit_weight", type=float, default=0.55)
parser.add_argument("--aus_yaw", type=float, default=1.57)
parser.add_argument("--aus_yaw_candidates", type=int, default=1)
parser.add_argument("--aus_yaw_span", type=float, default=0.0)
parser.add_argument("--aus_yaw_gain_weight", type=float, default=0.35)
parser.add_argument("--aus_yaw_strip_length", type=float, default=35.0)
parser.add_argument("--aus_yaw_strip_width", type=float, default=10.0)
parser.add_argument("--aus_roll", type=float, default=0.0)
parser.add_argument("--aus_roll_candidates", type=int, default=1)
parser.add_argument("--aus_roll_span", type=float, default=0.0)
parser.add_argument("--aus_pose_score_samples", type=int, default=9)
parser.add_argument("--aus_rh_horizon", type=int, default=4)
parser.add_argument("--aus_rh_branch_factor", type=int, default=8)
parser.add_argument("--aus_rh_beam_width", type=int, default=24)
parser.add_argument("--aus_rh_step_radius", type=float, default=45.0)
parser.add_argument("--aus_rh_path_length_weight", type=float, default=0.012)
parser.add_argument("--aus_rh_overlap_weight", type=float, default=0.9)
parser.add_argument("--split_file", type=str, default=str(DEFAULT_SPLIT_FILE))
parser.add_argument("--split", type=str, default=None, help="Patient split name, for example train or test.")
parser.add_argument("--patient_ids", type=str, default=None, help="Comma-separated patient id override.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
resolved_patient_ids = resolve_patient_ids(args_cli.patient_ids, args_cli.split, args_cli.split_file)
apply_patient_ids_env(resolved_patient_ids)

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import tqdm
import wandb

import isaaclab_tasks  # noqa: F401
import spinal_surgery  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from ruamel.yaml import YAML
from spinal_surgery import PACKAGE_DIR
from spinal_surgery.lab.controllers.heuristic_reconstruction import HeuristicReconstruction
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer

from trajectory_generation.anatomy_prior import (
    load_prior,
    prior_covered_mass,
)
from trajectory_generation.active_us_slam_planner import ActiveUSSLAMGoalPlanner, RecedingHorizonActiveUSSLAMGoalPlanner
from trajectory_generation.online_nbv_planner import OnlineNBVGoalPlanner
from trajectory_generation.patient_conditioned_prior import load_patient_prior_predictor
from trajectory_generation.registration_prior import RegistrationPriorPredictor
from trajectory_generation.view_gain_prior import load_view_gain_predictor


class RandomGoalPlanner:
    """Uniform command-space goal sampler inside the approximate target-organ range."""

    def __init__(
        self,
        human_pos_2d_min: torch.Tensor,
        human_pos_2d_max: torch.Tensor,
        replan_interval: int,
        yaw_min: float,
        yaw_max: float,
        device: str,
    ):
        self.human_pos_2d_min = human_pos_2d_min.to(device)
        self.human_pos_2d_max = human_pos_2d_max.to(device)
        self.replan_interval = max(1, int(replan_interval))
        self.yaw_min = float(yaw_min)
        self.yaw_max = float(yaw_max)
        self.device = device
        self.cached_goal: torch.Tensor | None = None

    def reset(self) -> None:
        self.cached_goal = None

    def goal(self, cur_cmd_state: torch.Tensor, step: int) -> torch.Tensor:
        if self.cached_goal is not None and step % self.replan_interval != 0:
            return self.cached_goal
        num_envs = cur_cmd_state.shape[0]
        rand_xz = torch.rand((num_envs, 2), device=self.device)
        xz = self.human_pos_2d_min + rand_xz * (self.human_pos_2d_max - self.human_pos_2d_min)
        yaw = torch.empty((num_envs, 1), device=self.device).uniform_(self.yaw_min, self.yaw_max)
        self.cached_goal = torch.cat([xz, yaw], dim=-1).detach().cpu()
        return self.cached_goal


def _expert_goal_pose(action_helper: HeuristicReconstruction, step: int) -> torch.Tensor:
    way_point_index = 0
    for index in range(len(action_helper.switch_steps) - 1):
        if step >= action_helper.switch_steps[index]:
            way_point_index = index
    return action_helper.way_points[way_point_index]


def _normalize_us_to_uint8(us_img: torch.Tensor) -> torch.Tensor:
    """Convert SonoGym US tensor to display/storage images per environment."""
    img = us_img[:, :, :, 0].detach()
    flat = img.reshape(img.shape[0], -1)
    img_min = flat.min(dim=1).values.reshape(-1, 1, 1)
    img_max = flat.max(dim=1).values.reshape(-1, 1, 1)
    img = (img - img_min) / (img_max - img_min + 1e-6)
    return (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu()


def _build_us_slicer(task_env, sim_mode: str) -> USSlicer:
    label_convert_map = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", "r"))
    us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", "r"))
    us_generative_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_generative_cfg.yaml", "r"))

    rec = task_env.surface_reconstructor
    label_maps = [label_map.detach().cpu().numpy() for label_map in rec.label_maps]
    ct_maps = [ct_map.detach().cpu().numpy() for ct_map in rec.ct_maps]

    return USSlicer(
        us_cfg,
        label_maps,
        ct_maps,
        task_env.sim_cfg["if_use_ct"],
        rec.human_list,
        task_env.scene.num_envs,
        task_env.sim_cfg["patient_xz_range"],
        task_env.sim_cfg["patient_xz_init_range"][0],
        task_env.sim.device,
        label_convert_map,
        us_cfg["image_size"],
        us_cfg["resolution"],
        img_thickness=1,
        visualize=False,
        sim_mode=sim_mode,
        us_generative_cfg=us_generative_cfg,
    )


def _reconstructed_volume_voxels(task_env) -> torch.Tensor:
    return task_env.surface_reconstructor.human_rec_volume.sum(dim=(1, 2, 3))


def _patient_ids_for_envs(task_env) -> list[str]:
    rec = task_env.surface_reconstructor
    human_ids = [os.path.basename(path.rstrip("/")) for path in rec.human_list]
    env_to_human = rec.env_to_human_inds.detach().cpu().tolist()
    return [human_ids[int(index)] for index in env_to_human]


def main() -> None:
    wandb.init(mode="disabled")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    task_env = env.unwrapped
    us_slicer = None if args_cli.no_save_us_images else _build_us_slicer(task_env, args_cli.us_sim_mode)
    anatomy_prior = load_prior(args_cli.anatomy_prior, task_env.sim.device) if args_cli.anatomy_prior else None
    patient_prior_predictor = (
        load_patient_prior_predictor(args_cli.patient_prior_model, task_env.sim.device)
        if args_cli.patient_prior_model
        else None
    )
    view_gain_predictor = (
        load_view_gain_predictor(args_cli.view_gain_model, task_env.sim.device)
        if args_cli.view_gain_model
        else None
    )
    prior_volume = anatomy_prior["prior_volume"].float() if anatomy_prior is not None else None
    if prior_volume is None and patient_prior_predictor is not None:
        prior_volume = patient_prior_predictor.global_prior.float()
    if args_cli.registration_prior:
        if prior_volume is None:
            raise KeyError("--registration_prior requires --anatomy_prior or a prior volume.")
        patient_prior_predictor = RegistrationPriorPredictor(prior_volume.float())

    human_pos_2d_min = task_env.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]] - args_cli.square_size
    human_pos_2d_max = task_env.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]] + args_cli.square_size

    action_helper = HeuristicReconstruction(
        max_action=task_env.max_action,
        action_scale=task_env.action_scale,
        human_pos_2d_min=human_pos_2d_min,
        human_pos_2d_max=human_pos_2d_max,
        num_sections=2,
        total_steps=args_cli.trajectory_length,
        ratio=[0.05, 0.05, 0.05, 0.0],
        device=task_env.sim.device,
    )
    if args_cli.goal_source == "nbv" and prior_volume is None:
        raise KeyError("NBV trajectory generation requires --anatomy_prior.")
    goal_planner = None
    if args_cli.goal_source == "nbv":
        goal_planner = OnlineNBVGoalPlanner(
            task_env,
            prior_volume,
            args_cli.nbv_replan_interval,
            args_cli.nbv_reach_radius,
            args_cli.nbv_observation_radius,
            args_cli.nbv_distance_weight,
            args_cli.nbv_visit_weight,
            args_cli.nbv_yaw,
        )
    elif args_cli.goal_source == "aus_slam":
        goal_planner = ActiveUSSLAMGoalPlanner(
            task_env,
            prior_volume,
            args_cli.aus_replan_interval,
            args_cli.aus_reach_radius,
            args_cli.aus_observation_radius,
            args_cli.aus_coverage_weight,
            args_cli.aus_uncertainty_weight,
            args_cli.aus_frontier_weight,
            args_cli.aus_prior_weight,
            args_cli.aus_distance_weight,
            args_cli.aus_revisit_weight,
            args_cli.aus_yaw,
            args_cli.aus_yaw_candidates,
            args_cli.aus_yaw_span,
            args_cli.aus_yaw_gain_weight,
            args_cli.aus_yaw_strip_length,
            args_cli.aus_yaw_strip_width,
            args_cli.aus_roll,
            args_cli.aus_roll_candidates,
            args_cli.aus_roll_span,
            args_cli.aus_pose_score_samples,
            patient_prior_predictor,
            args_cli.patient_prior_update_interval,
            view_gain_predictor,
        )
    elif args_cli.goal_source == "aus_rh_slam":
        goal_planner = RecedingHorizonActiveUSSLAMGoalPlanner(
            task_env,
            prior_volume,
            args_cli.aus_replan_interval,
            args_cli.aus_reach_radius,
            args_cli.aus_observation_radius,
            args_cli.aus_coverage_weight,
            args_cli.aus_uncertainty_weight,
            args_cli.aus_frontier_weight,
            args_cli.aus_prior_weight,
            args_cli.aus_distance_weight,
            args_cli.aus_revisit_weight,
            args_cli.aus_yaw,
            args_cli.aus_rh_horizon,
            args_cli.aus_rh_branch_factor,
            args_cli.aus_rh_beam_width,
            args_cli.aus_rh_step_radius,
            args_cli.aus_rh_path_length_weight,
            args_cli.aus_rh_overlap_weight,
            args_cli.aus_yaw_candidates,
            args_cli.aus_yaw_span,
            args_cli.aus_yaw_gain_weight,
            args_cli.aus_yaw_strip_length,
            args_cli.aus_yaw_strip_width,
            args_cli.aus_roll,
            args_cli.aus_roll_candidates,
            args_cli.aus_roll_span,
            args_cli.aus_pose_score_samples,
            patient_prior_predictor,
            args_cli.patient_prior_update_interval,
            view_gain_predictor,
        )
    elif args_cli.goal_source == "random":
        goal_planner = RandomGoalPlanner(
            human_pos_2d_min,
            human_pos_2d_max,
            args_cli.random_replan_interval,
            args_cli.random_yaw_min,
            args_cli.random_yaw_max,
            task_env.sim.device,
        )

    cmd_state_batches = []
    goal_batches = []
    action_batches = []
    us_image_batches = []
    us_ct_image_batches = []
    reconstructed_volume_batches = []
    delta_volume_batches = []
    prior_gain_reward_batches = []
    proxy_reward_batches = []
    coverage_batches = []
    trajectory_patient_ids = []
    rejected_coverages = []
    rejected_patient_ids = []

    attempt_batches = 0
    max_attempt_batches = max(math.ceil(args_cli.num_traj / args_cli.num_envs), int(args_cli.max_attempt_batches))
    pbar = tqdm.tqdm(total=args_cli.num_traj, desc="accepted-trajectories")
    while len(coverage_batches) < args_cli.num_traj and attempt_batches < max_attempt_batches:
        batch_index = attempt_batches
        attempt_batches += 1
        _, info = env.reset()
        if goal_planner is not None:
            goal_planner.reset()
        patient_ids_for_envs = _patient_ids_for_envs(task_env)

        cmd_states = []
        goals = []
        actions_list = []
        us_images = []
        us_ct_images = []
        reconstructed_volumes = []
        delta_volumes = []
        prior_gain_rewards = []
        proxy_rewards = []

        for step in range(args_cli.trajectory_length):
            if args_cli.goal_source == "expert":
                goal_cmd_pose = _expert_goal_pose(action_helper, step).to(task_env.sim.device)
                actions = action_helper.get_action(info, step)
            else:
                goal_cmd_pose = goal_planner.goal(info["cur_cmd_state"], step).to(task_env.sim.device)
                actions = action_helper.get_action_given_goal(info, goal_cmd_pose)
            volume_before = _reconstructed_volume_voxels(task_env)

            cmd_states.append(info["cur_cmd_state"].detach().cpu())
            goals.append(goal_cmd_pose.detach().cpu())
            actions_list.append(actions.detach().cpu())
            if us_slicer is not None:
                us_slicer.slice_US(
                    task_env.world_to_human_pos,
                    task_env.world_to_human_rot,
                    task_env.US_ee_pose_w[:, 0:3],
                    task_env.US_ee_pose_w[:, 3:7],
                )
                us_images.append(_normalize_us_to_uint8(us_slicer.us_img_tensor))
                us_ct_images.append(
                    task_env.surface_reconstructor.ct_img_tensor[:, :, :, 0].detach().cpu().to(torch.float16)
                )
            if prior_volume is not None:
                task_env.surface_reconstructor.slice_label_img(
                    task_env.world_to_human_pos,
                    task_env.world_to_human_rot,
                    task_env.US_ee_pose_w[:, 0:3],
                    task_env.US_ee_pose_w[:, 3:7],
                )
            if prior_volume is not None:
                prior_mass_before = prior_covered_mass(task_env.surface_reconstructor, prior_volume)
            else:
                prior_mass_before = None

            _, _, terminated, truncated, info = env.step(actions)
            volume_after = _reconstructed_volume_voxels(task_env)
            delta_volume = (volume_after - volume_before).clamp_min(0.0)
            if prior_mass_before is not None:
                prior_delta = (
                    prior_covered_mass(task_env.surface_reconstructor, prior_volume) - prior_mass_before
                ).clamp_min(0.0)
                prior_gain_rewards.append(prior_delta.detach().cpu())
            reconstructed_volumes.append(volume_after.detach().cpu())
            delta_volumes.append(delta_volume.detach().cpu())
            proxy_rewards.append(delta_volume.detach().cpu())
            if torch.any(torch.logical_or(terminated, truncated)).item() and step < args_cli.trajectory_length - 1:
                print(
                    f"[WARN] batch {batch_index} ended at step {step + 1}; "
                    f"requested length was {args_cli.trajectory_length}."
                )
                break

        cmd_state_batch = torch.stack(cmd_states, dim=1)
        goal_batch = torch.stack(goals, dim=1)
        action_batch = torch.stack(actions_list, dim=1)
        reconstructed_volume_batch = torch.stack(reconstructed_volumes, dim=1)
        delta_volume_batch = torch.stack(delta_volumes, dim=1)
        prior_gain_reward_batch = torch.stack(prior_gain_rewards, dim=1) if prior_gain_rewards else None
        proxy_reward_batch = torch.stack(proxy_rewards, dim=1)
        final_coverage = task_env.surface_reconstructor.get_converage_ratio().detach().cpu()

        us_image_batch = torch.stack(us_images, dim=1) if us_images else None
        us_ct_image_batch = torch.stack(us_ct_images, dim=1) if us_ct_images else None
        for env_index, coverage in enumerate(final_coverage.tolist()):
            if len(coverage_batches) >= args_cli.num_traj:
                break
            patient_id = patient_ids_for_envs[env_index]
            if float(coverage) < args_cli.min_final_coverage:
                rejected_coverages.append(float(coverage))
                rejected_patient_ids.append(patient_id)
                continue
            cmd_state_batches.append(cmd_state_batch[env_index : env_index + 1])
            goal_batches.append(goal_batch[env_index : env_index + 1])
            action_batches.append(action_batch[env_index : env_index + 1])
            reconstructed_volume_batches.append(reconstructed_volume_batch[env_index : env_index + 1])
            delta_volume_batches.append(delta_volume_batch[env_index : env_index + 1])
            if prior_gain_reward_batch is not None:
                prior_gain_reward_batches.append(prior_gain_reward_batch[env_index : env_index + 1])
            proxy_reward_batches.append(proxy_reward_batch[env_index : env_index + 1])
            if us_image_batch is not None:
                us_image_batches.append(us_image_batch[env_index : env_index + 1])
            if us_ct_image_batch is not None:
                us_ct_image_batches.append(us_ct_image_batch[env_index : env_index + 1])
            coverage_batches.append(final_coverage[env_index : env_index + 1])
            trajectory_patient_ids.append(patient_id)
            pbar.update(1)

    pbar.close()
    if len(coverage_batches) < args_cli.num_traj:
        raise RuntimeError(
            f"Only accepted {len(coverage_batches)} / {args_cli.num_traj} trajectories after "
            f"{attempt_batches} attempt batches. Lower --min_final_coverage or increase --max_attempt_batches."
        )

    cmd_state = torch.cat(cmd_state_batches, dim=0)[: args_cli.num_traj]
    goal_cmd_pose = torch.cat(goal_batches, dim=0)[: args_cli.num_traj]
    action = torch.cat(action_batches, dim=0)[: args_cli.num_traj]
    reconstructed_volume_voxels = torch.cat(reconstructed_volume_batches, dim=0)[: args_cli.num_traj]
    delta_reconstructed_volume_voxels = torch.cat(delta_volume_batches, dim=0)[: args_cli.num_traj]
    prior_gain_reward = (
        torch.cat(prior_gain_reward_batches, dim=0)[: args_cli.num_traj] if prior_gain_reward_batches else None
    )
    proxy_reward = torch.cat(proxy_reward_batches, dim=0)[: args_cli.num_traj]
    us_image = None
    us_ct_image = None
    if us_image_batches:
        us_image = torch.cat(us_image_batches, dim=0)[: args_cli.num_traj]
    if us_ct_image_batches:
        us_ct_image = torch.cat(us_ct_image_batches, dim=0)[: args_cli.num_traj]
    final_coverage = torch.cat(coverage_batches, dim=0)[: args_cli.num_traj]
    voxel_volume_m3 = float(task_env.surface_reconstructor.volume_res) ** 3

    output = {
        "task": args_cli.task,
        "kind": f"{args_cli.goal_source}_target_organ_trajectory",
        "split": args_cli.split,
        "split_file": str(args_cli.split_file),
        "patient_ids": [os.path.basename(path.rstrip("/")) for path in task_env.surface_reconstructor.human_list],
        "trajectory_patient_ids": trajectory_patient_ids,
        "num_traj": int(args_cli.num_traj),
        "trajectory_length": int(cmd_state.shape[1]),
        "expert_trajectory_length": int(cmd_state.shape[1]),
        "cmd_state": cmd_state,
        "goal_cmd_pose": goal_cmd_pose,
        "action": action,
        "reconstructed_volume_voxels": reconstructed_volume_voxels,
        "delta_reconstructed_volume_voxels": delta_reconstructed_volume_voxels,
        "reconstructed_volume": reconstructed_volume_voxels * voxel_volume_m3,
        "delta_reconstructed_volume": delta_reconstructed_volume_voxels * voxel_volume_m3,
        "prior_gain_reward": prior_gain_reward,
        "proxy_reward": proxy_reward,
        "final_coverage": final_coverage,
        "mean_final_coverage": float(final_coverage.mean().item()),
        "human_pos_2d_min": human_pos_2d_min.detach().cpu(),
        "human_pos_2d_max": human_pos_2d_max.detach().cpu(),
        "params": {
            "goal_source": args_cli.goal_source,
            "min_final_coverage": float(args_cli.min_final_coverage),
            "max_attempt_batches": int(args_cli.max_attempt_batches),
            "attempt_batches": int(attempt_batches),
            "accepted_trajectories": int(len(coverage_batches)),
            "rejected_trajectories": int(len(rejected_coverages)),
            "rejected_coverages": rejected_coverages,
            "rejected_patient_ids": rejected_patient_ids,
            "goal_pose_dim": int(goal_cmd_pose.shape[-1]),
            "aus_pose_model": (
                "registration_adaptive_xz_and_view_gain_se3"
                if args_cli.registration_prior and goal_cmd_pose.shape[-1] == 4 and args_cli.view_gain_model
                else "registration_adaptive_xz_and_surface_constrained_se3"
                if args_cli.registration_prior and goal_cmd_pose.shape[-1] == 4
                else "registration_adaptive_xz"
                if args_cli.registration_prior
                else "network_adaptive_xz_and_view_gain_se3"
                if goal_cmd_pose.shape[-1] == 4 and args_cli.patient_prior_model and args_cli.view_gain_model
                else "network_adaptive_xz_and_surface_constrained_se3"
                if goal_cmd_pose.shape[-1] == 4 and args_cli.patient_prior_model
                else "surface_constrained_se3"
                if goal_cmd_pose.shape[-1] == 4
                else "xz_yaw"
            ),
            "random_replan_interval": int(args_cli.random_replan_interval),
            "random_yaw_min": float(args_cli.random_yaw_min),
            "random_yaw_max": float(args_cli.random_yaw_max),
            "nbv_replan_interval": int(args_cli.nbv_replan_interval),
            "nbv_reach_radius": float(args_cli.nbv_reach_radius),
            "nbv_observation_radius": float(args_cli.nbv_observation_radius),
            "nbv_distance_weight": float(args_cli.nbv_distance_weight),
            "nbv_visit_weight": float(args_cli.nbv_visit_weight),
            "nbv_yaw": float(args_cli.nbv_yaw),
            "aus_replan_interval": int(args_cli.aus_replan_interval),
            "aus_reach_radius": float(args_cli.aus_reach_radius),
            "aus_observation_radius": float(args_cli.aus_observation_radius),
            "aus_coverage_weight": float(args_cli.aus_coverage_weight),
            "aus_uncertainty_weight": float(args_cli.aus_uncertainty_weight),
            "aus_frontier_weight": float(args_cli.aus_frontier_weight),
            "aus_prior_weight": float(args_cli.aus_prior_weight),
            "aus_distance_weight": float(args_cli.aus_distance_weight),
            "aus_revisit_weight": float(args_cli.aus_revisit_weight),
            "aus_yaw": float(args_cli.aus_yaw),
            "aus_yaw_candidates": int(args_cli.aus_yaw_candidates),
            "aus_yaw_span": float(args_cli.aus_yaw_span),
            "aus_yaw_gain_weight": float(args_cli.aus_yaw_gain_weight),
            "aus_yaw_strip_length": float(args_cli.aus_yaw_strip_length),
            "aus_yaw_strip_width": float(args_cli.aus_yaw_strip_width),
            "aus_roll": float(args_cli.aus_roll),
            "aus_roll_candidates": int(args_cli.aus_roll_candidates),
            "aus_roll_span": float(args_cli.aus_roll_span),
            "aus_pose_score_samples": int(args_cli.aus_pose_score_samples),
            "aus_rh_horizon": int(args_cli.aus_rh_horizon),
            "aus_rh_branch_factor": int(args_cli.aus_rh_branch_factor),
            "aus_rh_beam_width": int(args_cli.aus_rh_beam_width),
            "aus_rh_step_radius": float(args_cli.aus_rh_step_radius),
            "aus_rh_path_length_weight": float(args_cli.aus_rh_path_length_weight),
            "aus_rh_overlap_weight": float(args_cli.aus_rh_overlap_weight),
            "save_us_images": not args_cli.no_save_us_images,
            "us_sim_mode": args_cli.us_sim_mode,
            "us_image_source": "SonoGym USSlicer.us_img_tensor",
            "us_image_dtype": "uint8",
            "proxy_reward": "r_t = V_t - V_{t-1}, V is reconstructed target-structure voxel count",
            "anatomy_prior": args_cli.anatomy_prior,
            "patient_prior_model": args_cli.patient_prior_model,
            "registration_prior": bool(args_cli.registration_prior),
            "view_gain_model": args_cli.view_gain_model,
            "patient_prior_update_interval": int(args_cli.patient_prior_update_interval),
            "prior_gain_reward": "delta sum of anatomy prior probability on newly reconstructed target voxels",
            "voxel_volume_m3": voxel_volume_m3,
        },
    }
    if anatomy_prior is not None:
        output["anatomy_prior"] = {
            key: value
            for key, value in anatomy_prior.items()
            if key != "prior_volume"
        }
        output["anatomy_prior_volume"] = anatomy_prior["prior_volume"].detach().cpu()
    if us_image is not None:
        output["us_image"] = us_image
    if us_ct_image is not None:
        output["us_ct_image"] = us_ct_image

    output_dir = os.path.dirname(args_cli.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(output, args_cli.output)
    print(f"[RESULT] saved {cmd_state.shape[0]} trajectories to {args_cli.output}")
    print(f"[RESULT] trajectory_length={cmd_state.shape[1]}")
    if us_image is not None:
        print(f"[RESULT] us_image_shape={tuple(us_image.shape)}")
    if us_ct_image is not None:
        print(f"[RESULT] us_ct_image_shape={tuple(us_ct_image.shape)}")
    print(f"[RESULT] proxy_reward_shape={tuple(proxy_reward.shape)}")
    if prior_gain_reward is not None:
        print(f"[RESULT] prior_gain_reward_shape={tuple(prior_gain_reward.shape)}")
    print(f"[RESULT] proxy_reward_sum_mean={proxy_reward.sum(dim=1).mean().item():.4f}")
    print(f"[RESULT] mean_final_coverage={output['mean_final_coverage']:.4f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

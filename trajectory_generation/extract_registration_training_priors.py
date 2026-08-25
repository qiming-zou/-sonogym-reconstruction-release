"""Extract registration-prior snapshots for training patients.

The registration prior is online: it needs a sparse reconstruction produced by
executing the probe. This script runs one rollout per patient, registers the
global anatomy template to the current sparse reconstruction at selected steps,
and saves the resulting patient-specific priors.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajectory_generation.patient_splits import DEFAULT_SPLIT_FILE, apply_patient_ids_env, resolve_patient_ids

from isaaclab.app import AppLauncher


def _parse_snapshot_steps(value: str, max_steps: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(1, max_steps + 1))
    steps = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not steps:
        raise ValueError("--snapshot_steps must contain at least one step or `all`.")
    invalid = [step for step in steps if step < 1 or step > max_steps]
    if invalid:
        raise ValueError(f"Snapshot steps must be in [1, {max_steps}], got {invalid}.")
    return steps


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-robot-US-reconstruction-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--snapshot_steps", type=str, default="50,100,150,200,300,400,500")
parser.add_argument("--anatomy_prior", type=str, default="artifacts/checkpoints/anatomy_prior_l4_train.pt")
parser.add_argument("--output_dir", type=str, default="artifacts/priors/registration_train")
parser.add_argument("--merged_output", type=str, default="artifacts/priors/registration_train_all.pt")
parser.add_argument("--split_file", type=str, default=str(DEFAULT_SPLIT_FILE))
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--patient_ids", type=str, default=None)
parser.add_argument(
    "--goal_source",
    choices=("random", "expert", "aus_slam", "aus_rh_slam"),
    default="random",
    help="Trajectory source used to create the sparse reconstruction before registration.",
)
parser.add_argument("--square_size", type=float, default=25.0)
parser.add_argument("--random_replan_interval", type=int, default=20)
parser.add_argument("--random_yaw_min", type=float, default=0.0)
parser.add_argument("--random_yaw_max", type=float, default=1.57)
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
parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _patient_output_path(patient_id: str) -> Path:
    return Path(args_cli.output_dir) / f"registration_prior_{patient_id}.pt"


def _run_parent(patient_ids: list[str]) -> int:
    Path(args_cli.output_dir).mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    for patient_id in patient_ids:
        command = [
            sys.executable,
            str(script),
            "--worker",
            "--patient_ids",
            patient_id,
            "--task",
            args_cli.task,
            "--num_envs",
            "1",
            "--steps",
            str(args_cli.steps),
            "--snapshot_steps",
            args_cli.snapshot_steps,
            "--anatomy_prior",
            args_cli.anatomy_prior,
            "--output_dir",
            args_cli.output_dir,
            "--split_file",
            args_cli.split_file,
            "--goal_source",
            args_cli.goal_source,
            "--square_size",
            str(args_cli.square_size),
            "--random_replan_interval",
            str(args_cli.random_replan_interval),
            "--random_yaw_min",
            str(args_cli.random_yaw_min),
            "--random_yaw_max",
            str(args_cli.random_yaw_max),
            "--aus_replan_interval",
            str(args_cli.aus_replan_interval),
            "--aus_reach_radius",
            str(args_cli.aus_reach_radius),
            "--aus_observation_radius",
            str(args_cli.aus_observation_radius),
            "--aus_coverage_weight",
            str(args_cli.aus_coverage_weight),
            "--aus_uncertainty_weight",
            str(args_cli.aus_uncertainty_weight),
            "--aus_frontier_weight",
            str(args_cli.aus_frontier_weight),
            "--aus_prior_weight",
            str(args_cli.aus_prior_weight),
            "--aus_distance_weight",
            str(args_cli.aus_distance_weight),
            "--aus_revisit_weight",
            str(args_cli.aus_revisit_weight),
            "--aus_yaw",
            str(args_cli.aus_yaw),
            "--aus_yaw_candidates",
            str(args_cli.aus_yaw_candidates),
            "--aus_yaw_span",
            str(args_cli.aus_yaw_span),
            "--aus_yaw_gain_weight",
            str(args_cli.aus_yaw_gain_weight),
            "--aus_yaw_strip_length",
            str(args_cli.aus_yaw_strip_length),
            "--aus_yaw_strip_width",
            str(args_cli.aus_yaw_strip_width),
            "--aus_roll",
            str(args_cli.aus_roll),
            "--aus_roll_candidates",
            str(args_cli.aus_roll_candidates),
            "--aus_roll_span",
            str(args_cli.aus_roll_span),
            "--aus_pose_score_samples",
            str(args_cli.aus_pose_score_samples),
            "--aus_rh_horizon",
            str(args_cli.aus_rh_horizon),
            "--aus_rh_branch_factor",
            str(args_cli.aus_rh_branch_factor),
            "--aus_rh_beam_width",
            str(args_cli.aus_rh_beam_width),
            "--aus_rh_step_radius",
            str(args_cli.aus_rh_step_radius),
            "--aus_rh_path_length_weight",
            str(args_cli.aus_rh_path_length_weight),
            "--aus_rh_overlap_weight",
            str(args_cli.aus_rh_overlap_weight),
        ]
        if args_cli.headless:
            command.append("--headless")
        if args_cli.disable_fabric:
            command.append("--disable_fabric")
        if getattr(args_cli, "device", None):
            command.extend(["--device", str(args_cli.device)])
        _run(command)

    import torch

    outputs = []
    for patient_id in patient_ids:
        path = _patient_output_path(patient_id)
        outputs.append(torch.load(path, map_location="cpu", weights_only=False))
    merged = {
        "kind": "registration_training_prior_dataset",
        "patient_ids": patient_ids,
        "num_patients": len(patient_ids),
        "source_files": [str(_patient_output_path(patient_id)) for patient_id in patient_ids],
        "items": outputs,
    }
    merged_path = Path(args_cli.merged_output)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, merged_path)
    print(f"[RESULT] saved merged registration priors to {merged_path}")
    return 0


resolved_patient_ids = resolve_patient_ids(args_cli.patient_ids, args_cli.split, args_cli.split_file)
if not resolved_patient_ids:
    raise ValueError("No patient ids resolved. Pass --patient_ids or --split.")

if not args_cli.worker and len(resolved_patient_ids) > 1:
    raise SystemExit(_run_parent(resolved_patient_ids))

apply_patient_ids_env(resolved_patient_ids)
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import spinal_surgery  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from spinal_surgery.lab.controllers.heuristic_reconstruction import HeuristicReconstruction

from trajectory_generation.active_us_slam_planner import ActiveUSSLAMGoalPlanner, RecedingHorizonActiveUSSLAMGoalPlanner
from trajectory_generation.anatomy_prior import load_prior
from trajectory_generation.registration_prior import RegistrationPriorPredictor


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


def _patient_ids_for_envs(task_env) -> list[str]:
    rec = task_env.surface_reconstructor
    human_ids = [os.path.basename(path.rstrip("/")) for path in rec.human_list]
    env_to_human = rec.env_to_human_inds.detach().cpu().tolist()
    return [human_ids[int(index)] for index in env_to_human]


def _build_planner(task_env, prior_volume: torch.Tensor, registration_predictor: RegistrationPriorPredictor):
    human_pos_2d_min = task_env.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]] - args_cli.square_size
    human_pos_2d_max = task_env.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]] + args_cli.square_size
    if args_cli.goal_source == "random":
        return RandomGoalPlanner(
            human_pos_2d_min,
            human_pos_2d_max,
            args_cli.random_replan_interval,
            args_cli.random_yaw_min,
            args_cli.random_yaw_max,
            task_env.sim.device,
        )
    if args_cli.goal_source == "aus_slam":
        return ActiveUSSLAMGoalPlanner(
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
            registration_predictor,
            20,
            None,
        )
    if args_cli.goal_source == "aus_rh_slam":
        return RecedingHorizonActiveUSSLAMGoalPlanner(
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
            registration_predictor,
            20,
            None,
        )
    return None


def main() -> None:
    snapshot_steps = set(_parse_snapshot_steps(args_cli.snapshot_steps, args_cli.steps))
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    task_env = env.unwrapped
    anatomy_prior = load_prior(args_cli.anatomy_prior, task_env.sim.device)
    prior_volume = anatomy_prior["prior_volume"].float()
    registration_predictor = RegistrationPriorPredictor(prior_volume.float())

    human_pos_2d_min = task_env.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]] - args_cli.square_size
    human_pos_2d_max = task_env.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]] + args_cli.square_size
    action_helper = HeuristicReconstruction(
        max_action=task_env.max_action,
        action_scale=task_env.action_scale,
        human_pos_2d_min=human_pos_2d_min,
        human_pos_2d_max=human_pos_2d_max,
        num_sections=2,
        total_steps=args_cli.steps,
        ratio=[0.05, 0.05, 0.05, 0.0],
        device=task_env.sim.device,
    )
    planner = _build_planner(task_env, prior_volume, registration_predictor)

    _, info = env.reset()
    if planner is not None:
        planner.reset()
    patient_id = _patient_ids_for_envs(task_env)[0]
    priors = []
    sparse_reconstructions = []
    coverages = []
    observed_points = []
    blend_weights = []
    registration_stats = []
    saved_steps = []
    cmd_states = []
    goal_cmd_poses = []

    for step in range(args_cli.steps):
        if args_cli.goal_source == "expert":
            goal_cmd_pose = _expert_goal_pose(action_helper, step).to(task_env.sim.device)
            actions = action_helper.get_action(info, step)
        else:
            goal_cmd_pose = planner.goal(info["cur_cmd_state"], step).to(task_env.sim.device)
            actions = action_helper.get_action_given_goal(info, goal_cmd_pose)
        _, _, terminated, truncated, info = env.step(actions)
        current_step = step + 1
        if current_step in snapshot_steps:
            rec_volume = task_env.surface_reconstructor.human_rec_volume.detach().float()
            predicted_prior = registration_predictor.predict(rec_volume, [patient_id]).detach().cpu()
            coverage = task_env.surface_reconstructor.get_converage_ratio().detach().cpu()
            priors.append(predicted_prior[0])
            sparse_reconstructions.append(rec_volume[0].detach().cpu())
            coverages.append(coverage[0])
            observed_points.append(int((rec_volume[0] > 0).sum().detach().cpu().item()))
            blend_weights.append(registration_predictor.blend_weight(rec_volume)[0].detach().cpu())
            registration_stats.append(registration_predictor.last_registration_stats[0])
            saved_steps.append(current_step)
            cmd_states.append(info["cur_cmd_state"][0].detach().cpu())
            goal_cmd_poses.append(goal_cmd_pose[0].detach().cpu())
        if torch.any(torch.logical_or(terminated, truncated)).item() and current_step < args_cli.steps:
            print(f"[WARN] rollout ended at step {current_step}; requested {args_cli.steps}.")
            break

    output = {
        "kind": "registration_training_prior_snapshots",
        "patient_id": patient_id,
        "task": args_cli.task,
        "goal_source": args_cli.goal_source,
        "anatomy_prior": args_cli.anatomy_prior,
        "snapshot_steps": torch.tensor(saved_steps, dtype=torch.long),
        "registration_prior_volume": torch.stack(priors, dim=0) if priors else torch.empty(0),
        "sparse_reconstruction": torch.stack(sparse_reconstructions, dim=0) if sparse_reconstructions else torch.empty(0),
        "coverage": torch.stack(coverages, dim=0) if coverages else torch.empty(0),
        "observed_points": torch.tensor(observed_points, dtype=torch.long),
        "blend_weight": torch.stack(blend_weights, dim=0) if blend_weights else torch.empty(0),
        "registration_stats": registration_stats,
        "cmd_state": torch.stack(cmd_states, dim=0) if cmd_states else torch.empty(0),
        "goal_cmd_pose": torch.stack(goal_cmd_poses, dim=0) if goal_cmd_poses else torch.empty(0),
        "anatomy_prior_volume": anatomy_prior["prior_volume"].detach().cpu(),
        "params": {
            "steps": int(args_cli.steps),
            "snapshot_steps": sorted(snapshot_steps),
            "num_envs": int(args_cli.num_envs),
            "registration_method": "open3d_coarse_probreg_cpd_refine",
        },
    }
    output_path = _patient_output_path(patient_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"[RESULT] saved registration priors to {output_path}")
    print(f"[RESULT] patient_id={patient_id} snapshots={len(saved_steps)}")
    if coverages:
        print(f"[RESULT] final_snapshot_coverage={float(coverages[-1].item()):.4f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

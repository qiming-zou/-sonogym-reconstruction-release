"""Replay an optimized reconstruction trajectory in Isaac and report true coverage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajectory_generation.patient_splits import DEFAULT_SPLIT_FILE, apply_patient_ids_env, resolve_patient_ids

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay offline-RL reconstruction actions in Isaac.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="Isaac-robot-US-reconstruction-v0")
parser.add_argument("--trajectory", type=str, default="artifacts/trajectories/offline_rl_optimized_trajectory.pt")
parser.add_argument("--output_json", type=str, default="artifacts/trajectories/offline_rl_optimized_trajectory_eval.json")
parser.add_argument("--trajectory_index", type=int, default=0)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--mode", choices=("policy",), default="policy")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs for replay/evaluation.")
parser.add_argument("--split_file", type=str, default=str(DEFAULT_SPLIT_FILE))
parser.add_argument("--split", type=str, default=None, help="Patient split name, for example train or test.")
parser.add_argument("--patient_ids", type=str, default=None, help="Comma-separated patient id override.")
parser.add_argument(
    "--goal_source",
    choices=("nbv",),
    default="nbv",
    help="Goal source for goal-conditioned policies.",
)
parser.add_argument("--nbv_replan_interval", type=int, default=20)
parser.add_argument("--nbv_reach_radius", type=float, default=70.0)
parser.add_argument("--nbv_observation_radius", type=float, default=25.0)
parser.add_argument("--nbv_distance_weight", type=float, default=0.02)
parser.add_argument("--nbv_visit_weight", type=float, default=0.35)
parser.add_argument("--nbv_yaw", type=float, default=1.57)

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
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
from trajectory_generation.image_offline_policy import ImageCommandActor, Normalizer
from trajectory_generation.online_nbv_planner import OnlineNBVGoalPlanner


def _normalize_us_to_uint8(us_img: torch.Tensor) -> torch.Tensor:
    img = us_img[:, :, :, 0].detach()
    flat = img.reshape(img.shape[0], -1)
    img_min = flat.min(dim=1).values.reshape(-1, 1, 1)
    img_max = flat.max(dim=1).values.reshape(-1, 1, 1)
    img = (img - img_min) / (img_max - img_min + 1e-6)
    return (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)


def _build_us_slicer(task_env) -> USSlicer:
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
        sim_mode=data_sim_mode(task_env),
        us_generative_cfg=us_generative_cfg,
    )


def data_sim_mode(task_env) -> str:
    return getattr(task_env, "offline_replay_us_sim_mode", "net")


def image_key_from_state_mode(state_mode: str) -> str:
    if state_mode in ("us_image_goal_cmd", "us_image_cmd"):
        return "us_image"
    raise ValueError(f"Unsupported state_mode `{state_mode}`.")


def uses_command_state(state_mode: str) -> bool:
    return state_mode in ("us_image_goal_cmd", "us_image_cmd")


def uses_goal_command_state(state_mode: str) -> bool:
    return state_mode == "us_image_goal_cmd"


def _live_policy_image(
    task_env,
    us_slicer: USSlicer | None,
    image_key: str,
) -> torch.Tensor:
    if image_key != "us_image":
        raise ValueError(f"Unsupported policy image key `{image_key}`.")
    task_env.surface_reconstructor.slice_label_img(
        task_env.world_to_human_pos,
        task_env.world_to_human_rot,
        task_env.US_ee_pose_w[:, 0:3],
        task_env.US_ee_pose_w[:, 3:7],
    )
    if us_slicer is None:
        raise RuntimeError("USSlicer is required for live ultrasound policy replay.")
    us_slicer.slice_US(
        task_env.world_to_human_pos,
        task_env.world_to_human_rot,
        task_env.US_ee_pose_w[:, 0:3],
        task_env.US_ee_pose_w[:, 3:7],
    )
    us_image = _normalize_us_to_uint8(us_slicer.us_img_tensor).to(task_env.sim.device)
    return us_image.unsqueeze(1)


def _patient_ids_for_envs(task_env) -> list[str]:
    rec = task_env.surface_reconstructor
    human_ids = [os.path.basename(path.rstrip("/")) for path in rec.human_list]
    env_to_human = rec.env_to_human_inds.detach().cpu().tolist()
    return [human_ids[int(index)] for index in env_to_human]


def _first_done_steps(current_steps: list[int | None], done_tensor: torch.Tensor, step: int) -> list[int | None]:
    done_cpu = done_tensor.detach().cpu().bool().tolist()
    for env_index, is_done in enumerate(done_cpu):
        if is_done and current_steps[env_index] is None:
            current_steps[env_index] = step + 1
    return current_steps


def _live_command_state(
    info: dict,
    data: dict,
    step: int,
    trajectory_index: int,
    state_mode: str,
    goal_planner: OnlineNBVGoalPlanner | None,
    goal_source: str,
) -> torch.Tensor:
    cur_cmd_state = info["cur_cmd_state"].detach().cpu().float()
    if not uses_goal_command_state(state_mode):
        return cur_cmd_state
    if goal_source != "nbv":
        raise ValueError(f"Unsupported goal_source={goal_source!r}; only online NBV is retained.")
    if goal_planner is None:
        raise ValueError("NBV replay requires a goal planner.")
    goal = goal_planner.goal(cur_cmd_state.to(goal_planner.task_env.sim.device), step).float()
    goal_delta = goal - cur_cmd_state[:, : goal.shape[-1]]
    return torch.cat([cur_cmd_state, goal, goal_delta], dim=-1)


def main() -> None:
    wandb.init(mode="disabled")

    data = torch.load(args_cli.trajectory, map_location="cpu")
    actions = data["action"].float()
    if actions.ndim != 3:
        raise ValueError(f"Expected action tensor shape (N, T, A), got {tuple(actions.shape)}")
    if args_cli.trajectory_index >= actions.shape[0]:
        raise IndexError(f"trajectory_index={args_cli.trajectory_index} but only {actions.shape[0]} trajectories exist")

    action_sequence = actions[args_cli.trajectory_index]
    if args_cli.max_steps is not None:
        action_sequence = action_sequence[: args_cli.max_steps]

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    task_env = env.unwrapped
    task_env.offline_replay_us_sim_mode = data.get("training", {}).get("us_sim_mode", "net")

    action_norm = None
    cmd_norm = None
    state_mode = data.get("state_key", data.get("policy_state", {}).get("state_mode", "us_image"))
    image_key = data.get("image_key", data.get("policy_state", {}).get("image_key", image_key_from_state_mode(state_mode)))
    use_command = uses_command_state(state_mode)
    action_min = None
    action_max = None
    us_slicer = None
    prior_volume = data.get("anatomy_prior_volume")
    if prior_volume is not None:
        prior_volume = prior_volume.float().to(task_env.sim.device)
    goal_planner = None
    if "policy_state" not in data:
        raise KeyError("Trajectory file does not contain `policy_state`; rerun offline RL training first.")
    if not use_command:
        raise ValueError("Only command-conditioned policies are supported.")
    policy_state = data["policy_state"]
    action_limit = policy_state["action_limit"].to(task_env.sim.device)
    input_channels = int(policy_state.get("input_channels", 1))
    actor = ImageCommandActor(
        int(policy_state["action_dim"]),
        int(policy_state["cmd_state_dim"]),
        int(policy_state["hidden_dim"]),
        int(policy_state["feature_dim"]),
        action_limit,
        input_channels,
    ).to(task_env.sim.device)
    actor.load_state_dict({k: v.to(task_env.sim.device) for k, v in policy_state["actor_state_dict"].items()})
    actor.eval()
    norm = data["normalizers"]["action"]
    action_norm = Normalizer(norm["mean"].float(), norm["std"].float())
    cmd_norm_data = data["normalizers"]["cmd_state"]
    cmd_norm = Normalizer(cmd_norm_data["mean"].float(), cmd_norm_data["std"].float())
    bounds = data.get("action_bounds")
    if bounds is not None:
        action_min = bounds["min"].float().to(task_env.sim.device)
        action_max = bounds["max"].float().to(task_env.sim.device)
    us_slicer = _build_us_slicer(task_env)
    if args_cli.goal_source == "nbv" and uses_goal_command_state(state_mode):
        if prior_volume is None:
            raise KeyError("NBV goal planning requires `anatomy_prior_volume` in the trajectory file.")
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

    _, info = env.reset()
    patient_ids = _patient_ids_for_envs(task_env)
    coverage_trace = []
    coverage_trace_per_env = []
    reward_trace = []
    reward_trace_per_env = []
    terminated_steps = [None] * task_env.scene.num_envs
    truncated_steps = [None] * task_env.scene.num_envs

    for step, action in enumerate(tqdm.tqdm(action_sequence, desc=f"replay-{args_cli.mode}")):
        live_image = _live_policy_image(task_env, us_slicer, image_key)
        with torch.no_grad():
            live_cmd_raw = _live_command_state(
                info,
                data,
                step,
                args_cli.trajectory_index,
                state_mode,
                goal_planner,
                args_cli.goal_source,
            )
            live_cmd = cmd_norm.encode(live_cmd_raw).to(task_env.sim.device)
            action_encoded = actor(live_image, live_cmd, deterministic=True)
            action_batch = action_norm.decode(action_encoded.cpu()).to(task_env.sim.device)
            if action_min is not None and action_max is not None:
                action_batch = torch.clamp(action_batch, min=action_min, max=action_max)
        _, reward, terminated, truncated, info = env.step(action_batch)
        coverage_tensor = task_env.surface_reconstructor.get_converage_ratio().detach().cpu()
        reward_tensor = reward.detach().cpu()
        coverage_trace.append(float(coverage_tensor[0]))
        coverage_trace_per_env.append([float(value) for value in coverage_tensor.tolist()])
        reward_trace.append(float(reward_tensor[0]))
        reward_trace_per_env.append([float(value) for value in reward_tensor.tolist()])

        terminated_steps = _first_done_steps(terminated_steps, terminated, step)
        truncated_steps = _first_done_steps(truncated_steps, truncated, step)
        if torch.all(torch.logical_or(terminated, truncated)).item():
            break

    if coverage_trace_per_env:
        final_coverage_per_env = coverage_trace_per_env[-1]
    else:
        final_coverage_per_env = [
            float(value) for value in task_env.surface_reconstructor.get_converage_ratio().detach().cpu().tolist()
        ]
    final_coverage = sum(final_coverage_per_env) / max(1, len(final_coverage_per_env))
    max_coverage_per_env = [
        max(step_values[env_index] for step_values in coverage_trace_per_env)
        for env_index in range(len(final_coverage_per_env))
    ] if coverage_trace_per_env else final_coverage_per_env
    mean_reward_per_env = [
        sum(step_values[env_index] for step_values in reward_trace_per_env) / max(1, len(reward_trace_per_env))
        for env_index in range(len(final_coverage_per_env))
    ] if reward_trace_per_env else [0.0 for _ in final_coverage_per_env]
    env.close()

    result = {
        "trajectory_file": args_cli.trajectory,
        "kind": data.get("kind", "unknown"),
        "policy": data.get("policy", "unknown"),
        "state_key": data.get("state_key", "unknown"),
        "image_key": image_key,
        "split": args_cli.split,
        "split_file": str(args_cli.split_file),
        "status": "replayed_in_isaac",
        "mode": args_cli.mode,
        "goal_source": args_cli.goal_source,
        "nbv": {
            "replan_interval": int(args_cli.nbv_replan_interval),
            "reach_radius": float(args_cli.nbv_reach_radius),
            "observation_radius": float(args_cli.nbv_observation_radius),
            "distance_weight": float(args_cli.nbv_distance_weight),
            "visit_weight": float(args_cli.nbv_visit_weight),
            "yaw": float(args_cli.nbv_yaw),
        },
        "task": args_cli.task,
        "num_envs": int(args_cli.num_envs),
        "patient_ids": patient_ids,
        "trajectory_index": int(args_cli.trajectory_index),
        "max_steps": args_cli.max_steps,
        "requested_steps": int(actions.shape[1]),
        "replayed_steps": int(len(coverage_trace)),
        "replay_final_coverage": float(final_coverage),
        "replay_final_coverage_per_env": final_coverage_per_env,
        "replay_final_coverage_by_patient": {
            patient_id: float(coverage) for patient_id, coverage in zip(patient_ids, final_coverage_per_env)
        },
        "replay_max_coverage": float(sum(max_coverage_per_env) / max(1, len(max_coverage_per_env))),
        "replay_max_coverage_per_env": max_coverage_per_env,
        "replay_mean_reward": float(sum(reward_trace) / max(1, len(reward_trace))),
        "replay_mean_reward_per_env": mean_reward_per_env,
        "terminated_step": terminated_steps[0],
        "truncated_step": truncated_steps[0],
        "terminated_steps": terminated_steps,
        "truncated_steps": truncated_steps,
        "source_best_final_coverage": data.get("source_best_final_coverage"),
        "source_mean_final_coverage": data.get("source_mean_final_coverage"),
        "predicted_q_mean": float(data["predicted_q"].float().mean().item()) if "predicted_q" in data else None,
        "coverage_trace": coverage_trace,
        "coverage_trace_per_env": coverage_trace_per_env,
    }

    output_path = Path(args_cli.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"[RESULT] saved replay eval to {output_path}")
    print(f"[RESULT] replayed_steps={result['replayed_steps']}")
    print(f"[RESULT] replay_final_coverage_mean={result['replay_final_coverage']:.6f}")
    print(f"[RESULT] replay_max_coverage_mean={result['replay_max_coverage']:.6f}")
    print(f"[RESULT] replay_mean_reward={result['replay_mean_reward']:.6f}")
    print(f"[RESULT] patients={patient_ids}")
    print(f"[RESULT] replay_final_coverage_per_env={final_coverage_per_env}")
    print(f"[RESULT] terminated_steps={terminated_steps}")
    print(f"[RESULT] truncated_steps={truncated_steps}")


if __name__ == "__main__":
    main()
    simulation_app.close()

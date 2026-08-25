from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajectory_generation.patient_splits import DEFAULT_SPLIT_FILE, apply_patient_ids_env, resolve_patient_ids

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Run the heuristic reconstruction expert and export a reconstruction video."
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-robot-US-reconstruction-v0")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--capture_interval", type=int, default=5)
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--output_dir", type=str, default="artifacts/videos")
parser.add_argument("--thumbnail_dir", type=str, default="artifacts/thumbnails")
parser.add_argument(
    "--trajectory_source",
    choices=(
        "expert",
        "nbv",
        "policy",
        "stitching",
        "random",
        "expert_replay",
        "aus_slam",
        "aus_rh_slam",
        "aus_rh_replay",
    ),
    default="expert",
)
parser.add_argument("--trajectory", type=str, default="artifacts/trajectories/offline_rl_apo_us_goal_cmd_train.pt")
parser.add_argument("--trajectory_index", type=int, default=0)
parser.add_argument("--split_file", type=str, default=str(DEFAULT_SPLIT_FILE))
parser.add_argument("--split", type=str, default=None, help="Patient split name, for example train or test.")
parser.add_argument("--patient_ids", type=str, default=None, help="Comma-separated patient id override.")
parser.add_argument("--square_size", type=float, default=25.0)
parser.add_argument("--planner_total_steps", type=int, default=500)
parser.add_argument("--anatomy_prior", type=str, default="artifacts/checkpoints/anatomy_prior_l4_train.pt")
parser.add_argument("--patient_prior_model", type=str, default=None)
parser.add_argument("--view_gain_model", type=str, default=None)
parser.add_argument("--registration_prior", action="store_true")
parser.add_argument("--patient_prior_update_interval", type=int, default=20)
parser.add_argument("--visualize_patient_prior", action="store_true")
parser.add_argument("--prior_overlay_threshold", type=float, default=0.35)
parser.add_argument("--prior_overlay_max_points", type=int, default=1400)
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
parser.add_argument("--layout", choices=("site", "dashboard"), default="site")
parser.add_argument("--render_scene", action="store_true", help="Use Isaac rgb_array rendering for the left panel.")
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=(0.85, 1.05, 1.32),
    help="World-space camera eye for the rendered scene panel.",
)
parser.add_argument(
    "--camera_target",
    type=float,
    nargs=3,
    default=(0.20, -0.10, 0.82),
    help="World-space camera target for the rendered scene panel.",
)
parser.add_argument(
    "--scene_zoom",
    type=float,
    default=1.95,
    help="Additional center crop zoom for the rendered scene panel.",
)
parser.add_argument(
    "--scene_offset",
    type=float,
    nargs=2,
    default=(-0.10, 0.65),
    metavar=("X", "Y"),
    help="Crop offset for the rendered scene panel in normalized extra-crop units.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
resolved_patient_ids = resolve_patient_ids(args_cli.patient_ids, args_cli.split, args_cli.split_file)
apply_patient_ids_env(resolved_patient_ids)
if args_cli.layout == "site" and args_cli.render_scene:
    args_cli.enable_cameras = True
    driver_check_arg = "--/rtx/verifyDriverVersion/enabled=false"
    if driver_check_arg not in args_cli.kit_args:
        args_cli.kit_args = f"{args_cli.kit_args} {driver_check_arg}".strip()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import font_manager

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
CHINESE_FONT = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
CHINESE_FONT_BOLD = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import isaaclab_tasks  # noqa: F401
import spinal_surgery  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms, transform_points
from ruamel.yaml import YAML
from spinal_surgery import PACKAGE_DIR
from spinal_surgery.lab.controllers.heuristic_reconstruction import (
    HeuristicReconstruction,
)
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
from trajectory_generation.anatomy_prior import load_prior
from trajectory_generation.active_us_slam_planner import ActiveUSSLAMGoalPlanner, RecedingHorizonActiveUSSLAMGoalPlanner
from trajectory_generation.image_offline_policy import ImageCommandActor, Normalizer
from trajectory_generation.online_nbv_planner import OnlineNBVGoalPlanner
from trajectory_generation.patient_conditioned_prior import load_patient_prior_predictor
from trajectory_generation.registration_prior import RegistrationPriorPredictor
from trajectory_generation.view_gain_prior import load_view_gain_predictor

SITE_BACKGROUND = "#eaf6fb"
TRAJECTORY_LABELS = {
    "expert": "启发式专家轨迹",
    "nbv": "在线NBV轨迹",
    "policy": "SAC_BC测试轨迹",
    "stitching": "DiffStitch拼接轨迹",
    "random": "原始随机轨迹",
    "expert_replay": "专家轨迹回放",
    "aus_slam": "AUS-SLAM主动建图",
    "aus_rh_slam": "AUS-RH-SLAM短视界主动建图",
    "aus_rh_replay": "AUS-RH-SLAM测试轨迹回放",
}
TRAJECTORY_LABEL = TRAJECTORY_LABELS[args_cli.trajectory_source]


def _normalize_us_to_uint8(us_img: torch.Tensor) -> torch.Tensor:
    img = us_img[:, :, :, 0].detach()
    flat = img.reshape(img.shape[0], -1)
    img_min = flat.min(dim=1).values.reshape(-1, 1, 1)
    img_max = flat.max(dim=1).values.reshape(-1, 1, 1)
    img = (img - img_min) / (img_max - img_min + 1e-6)
    return (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)


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


def _uses_goal_command_state(state_mode: str) -> bool:
    return state_mode == "us_image_goal_cmd"


def _live_policy_image(task_env, us_slicer: USSlicer) -> torch.Tensor:
    task_env.surface_reconstructor.slice_label_img(
        task_env.world_to_human_pos,
        task_env.world_to_human_rot,
        task_env.US_ee_pose_w[:, 0:3],
        task_env.US_ee_pose_w[:, 3:7],
    )
    us_slicer.slice_US(
        task_env.world_to_human_pos,
        task_env.world_to_human_rot,
        task_env.US_ee_pose_w[:, 0:3],
        task_env.US_ee_pose_w[:, 3:7],
    )
    return _normalize_us_to_uint8(us_slicer.us_img_tensor).to(task_env.sim.device).unsqueeze(1)


def _live_policy_command_state(
    info: dict,
    state_mode: str,
    goal_planner: OnlineNBVGoalPlanner,
    step: int,
) -> torch.Tensor:
    cur_cmd_state = info["cur_cmd_state"].detach().cpu().float()
    if not _uses_goal_command_state(state_mode):
        return cur_cmd_state
    goal = goal_planner.goal(cur_cmd_state.to(goal_planner.task_env.sim.device), step).float()
    goal_delta = goal - cur_cmd_state[:, : goal.shape[-1]]
    return torch.cat([cur_cmd_state, goal, goal_delta], dim=-1)


def _load_policy_components(task_env, trajectory_path: str):
    data = torch.load(trajectory_path, map_location="cpu")
    if "policy_state" not in data:
        raise KeyError("Policy video requires a trajectory file with `policy_state`.")
    state_mode = data.get("state_key", data.get("policy_state", {}).get("state_mode", "us_image_goal_cmd"))
    policy_state = data["policy_state"]
    if state_mode not in ("us_image_goal_cmd", "us_image_cmd"):
        raise ValueError(f"Unsupported policy state mode for video: {state_mode}")
    prior_volume = data.get("anatomy_prior_volume")
    if prior_volume is None:
        raise KeyError("Policy video with online NBV goals requires `anatomy_prior_volume` in the trajectory file.")

    action_limit = policy_state["action_limit"].to(task_env.sim.device)
    actor = ImageCommandActor(
        int(policy_state["action_dim"]),
        int(policy_state["cmd_state_dim"]),
        int(policy_state["hidden_dim"]),
        int(policy_state["feature_dim"]),
        action_limit,
        int(policy_state.get("input_channels", 1)),
    ).to(task_env.sim.device)
    actor.load_state_dict({key: value.to(task_env.sim.device) for key, value in policy_state["actor_state_dict"].items()})
    actor.eval()

    action_norm_data = data["normalizers"]["action"]
    action_norm = Normalizer(action_norm_data["mean"].float(), action_norm_data["std"].float())
    cmd_norm_data = data["normalizers"]["cmd_state"]
    cmd_norm = Normalizer(cmd_norm_data["mean"].float(), cmd_norm_data["std"].float())

    bounds = data.get("action_bounds")
    action_min = bounds["min"].float().to(task_env.sim.device) if bounds is not None else None
    action_max = bounds["max"].float().to(task_env.sim.device) if bounds is not None else None
    us_slicer = _build_us_slicer(task_env, data.get("training", {}).get("us_sim_mode", "net"))
    nbv_planner = OnlineNBVGoalPlanner(
        task_env,
        prior_volume.float().to(task_env.sim.device),
        args_cli.nbv_replan_interval,
        args_cli.nbv_reach_radius,
        args_cli.nbv_observation_radius,
        args_cli.nbv_distance_weight,
        args_cli.nbv_visit_weight,
        args_cli.nbv_yaw,
    )
    return data, state_mode, actor, action_norm, cmd_norm, action_min, action_max, us_slicer, nbv_planner


def _load_saved_actions(trajectory_path: str, trajectory_index: int, source_name: str) -> tuple[dict, torch.Tensor]:
    data = torch.load(trajectory_path, map_location="cpu")
    if "action" not in data:
        raise KeyError(f"{source_name} video requires a trajectory file with `action`.")
    actions = data["action"].float()
    if actions.ndim != 3:
        raise ValueError(f"Expected action tensor shape (N, T, A), got {tuple(actions.shape)}")
    if trajectory_index >= actions.shape[0]:
        raise IndexError(f"trajectory_index={trajectory_index} but only {actions.shape[0]} trajectories exist.")
    return data, actions[trajectory_index]


def _vec3f_array(points: np.ndarray) -> Vt.Vec3fArray:
    return Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points])


def _make_preview_material(
    stage,
    path: str,
    color: tuple[float, float, float],
    roughness: float = 0.62,
    metallic: float = 0.0,
    specular: tuple[float, float, float] | None = None,
):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if specular is not None:
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*specular))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _set_scene_render_style(task_env) -> None:
    stage = task_env.sim.stage
    robot_color = Gf.Vec3f(0.78, 0.80, 0.78)
    robot_material = _make_preview_material(stage, "/World/Looks/RobotMatteWarmGray", tuple(robot_color), 0.55)
    probe_color = Gf.Vec3f(0.05, 0.06, 0.07)
    probe_material = _make_preview_material(
        stage,
        "/World/Looks/ProbeGlossyGraphite",
        tuple(probe_color),
        roughness=0.18,
        metallic=0.15,
        specular=(0.95, 0.98, 1.0),
    )
    human_color = Gf.Vec3f(0.54, 0.32, 0.30)
    human_material = _make_preview_material(
        stage,
        "/World/Looks/HumanContactWarmSkin",
        tuple(human_color),
        roughness=0.82,
        specular=(0.18, 0.12, 0.10),
    )
    floor_material = _make_preview_material(stage, "/World/Looks/CoolBlueGrayFloor", (0.62, 0.68, 0.72), 0.8)

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_path_lower = prim_path.lower()
        is_probe_visual = (
            "/lbr_link_7/" in prim_path
            or "link_ee" in prim_path_lower
            or "end_effector" in prim_path_lower
            or "probe" in prim_path_lower
            or "transducer" in prim_path_lower
            or "ultrasound" in prim_path_lower
        )
        is_robot_visual = (
            prim_path.startswith("/World/envs/env_0/Robot_US")
            or prim_path.startswith("/World/Template/Asset_0000")
            or "/lbr_link_" in prim_path
        )
        is_human_visual = prim_path.startswith("/World/envs/env_0/Human")
        if is_probe_visual and prim.IsA(UsdGeom.Gprim):
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr([probe_color])
            gprim.CreateDisplayOpacityAttr([1.0])
            UsdShade.MaterialBindingAPI(prim).Bind(probe_material)
        elif is_human_visual and prim.IsA(UsdGeom.Gprim):
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr([human_color])
            gprim.CreateDisplayOpacityAttr([1.0])
            UsdShade.MaterialBindingAPI(prim).Bind(human_material)
        elif is_robot_visual and prim.IsA(UsdGeom.Gprim):
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr([robot_color])
            gprim.CreateDisplayOpacityAttr([1.0])
            UsdShade.MaterialBindingAPI(prim).Bind(robot_material)
        elif prim_path.startswith("/World/defaultGroundPlane") and prim.IsA(UsdGeom.Gprim):
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.68, 0.72)])
            UsdShade.MaterialBindingAPI(prim).Bind(floor_material)

    light_prim = stage.GetPrimAtPath("/World/Light")
    if light_prim and light_prim.IsValid():
        color_attr = light_prim.GetAttribute("inputs:color")
        intensity_attr = light_prim.GetAttribute("inputs:intensity")
        if color_attr:
            color_attr.Set(Gf.Vec3f(0.72, 0.76, 0.82))
        if intensity_attr:
            intensity_attr.Set(1600.0)

    probe_light = UsdLux.SphereLight.Define(stage, "/World/ProbeHighlightLight")
    probe_light.CreateIntensityAttr(12500.0)
    probe_light.CreateRadiusAttr(0.12)
    probe_light.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.92))
    probe_light.AddTranslateOp().Set(Gf.Vec3d(0.45, 0.42, 1.28))

    surface_light = UsdLux.SphereLight.Define(stage, "/World/HumanSurfaceRakingLight")
    surface_light.CreateIntensityAttr(8500.0)
    surface_light.CreateRadiusAttr(0.09)
    surface_light.CreateColorAttr(Gf.Vec3f(1.0, 0.78, 0.66))
    surface_light.AddTranslateOp().Set(Gf.Vec3d(0.95, -0.40, 1.12))


def _set_target_anatomy_scene_overlay(task_env, max_points: int = 3500) -> None:
    stage = task_env.sim.stage

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.startswith("/World/envs/env_0/Human") and prim.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(prim).CreateDisplayOpacityAttr([1.0])

    points_np = _target_vertebra_world_points(task_env, max_points)

    target_points = UsdGeom.Points.Define(stage, "/World/envs/env_0/TargetVertebraPoints")
    target_points.CreatePointsAttr(_vec3f_array(points_np))
    target_points.CreateWidthsAttr(Vt.FloatArray([0.009] * points_np.shape[0]))
    target_points.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.78, 1.0)])
    target_points.CreateDisplayOpacityAttr([1.0])


def _target_vertebra_world_points(task_env, max_points: int = 1400) -> np.ndarray:
    vertebra_points = (
        torch.as_tensor(
            task_env.vertebra_viewer.vertebra_points_list[0],
            device=task_env.sim.device,
        ).float()
        * float(task_env.vertebra_viewer.res)
    )
    world_points = transform_points(
        vertebra_points.unsqueeze(0),
        task_env.world_to_human_pos[0:1],
        task_env.world_to_human_rot[0:1],
    )[0]
    points_np = world_points.detach().cpu().numpy()
    if points_np.shape[0] > max_points:
        points_np = points_np[:: int(np.ceil(points_np.shape[0] / max_points))]
    return points_np


def _reconstruction_world_points(task_env, max_points: int = 900) -> np.ndarray:
    rec = task_env.surface_reconstructor
    volume = rec.human_rec_volume[0].detach().cpu().numpy() > 0
    points = np.argwhere(volume)
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    if points.shape[0] > max_points:
        points = points[:: int(np.ceil(points.shape[0] / max_points))]

    corner = rec.human_rec_volume_corner[0].detach().cpu().numpy()
    human_points = points.astype(np.float32) * float(rec.volume_res) + corner.astype(np.float32)
    world_points = transform_points(
        torch.as_tensor(human_points, device=rec.device).unsqueeze(0),
        task_env.world_to_human_pos[0:1],
        task_env.world_to_human_rot[0:1],
    )[0]
    return world_points.detach().cpu().numpy()


def _patient_ids_for_envs(task_env) -> list[str]:
    rec = task_env.surface_reconstructor
    human_ids = [Path(path.rstrip("/")).name for path in rec.human_list]
    env_to_human = rec.env_to_human_inds.detach().cpu().tolist()
    return [human_ids[int(index)] for index in env_to_human]


def _estimated_patient_prior_volume(task_env, patient_prior_predictor) -> np.ndarray | None:
    if patient_prior_predictor is None:
        return None
    rec = task_env.surface_reconstructor
    with torch.inference_mode():
        prior = patient_prior_predictor.predict(
            rec.human_rec_volume.detach().float(),
            _patient_ids_for_envs(task_env),
        )
    return prior[0].detach().cpu().numpy().astype(np.float32)


def _prior_volume_points(
    prior_volume: np.ndarray | None,
    threshold: float,
    max_points: int,
) -> np.ndarray:
    if prior_volume is None or prior_volume.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    flat = prior_volume.reshape(-1)
    candidate = np.flatnonzero(flat >= float(threshold))
    if candidate.size == 0:
        top_k = min(max_points, flat.size)
        candidate = np.argpartition(flat, -top_k)[-top_k:]
    elif candidate.size > max_points:
        values = flat[candidate]
        top_idx = np.argpartition(values, -max_points)[-max_points:]
        candidate = candidate[top_idx]
    coords = np.column_stack(np.unravel_index(candidate, prior_volume.shape)).astype(np.float32)
    return coords


def _estimated_prior_world_points(
    task_env,
    prior_volume: np.ndarray | None,
    threshold: float,
    max_points: int,
) -> np.ndarray:
    points = _prior_volume_points(prior_volume, threshold, max_points)
    if points.size == 0:
        return points
    rec = task_env.surface_reconstructor
    corner = rec.human_rec_volume_corner[0].detach().cpu().numpy()
    human_points = points * float(rec.volume_res) + corner.astype(np.float32)
    world_points = transform_points(
        torch.as_tensor(human_points, device=rec.device).unsqueeze(0),
        task_env.world_to_human_pos[0:1],
        task_env.world_to_human_rot[0:1],
    )[0]
    return world_points.detach().cpu().numpy()


def _prior_xz_points(
    prior_volume: np.ndarray | None,
    threshold: float,
    max_points: int = 2200,
) -> np.ndarray:
    points = _prior_volume_points(prior_volume, threshold, max_points * 2)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    xz = np.unique(points[:, [0, 2]].astype(np.int32), axis=0).astype(np.float32)
    if xz.shape[0] > max_points:
        xz = xz[:: int(np.ceil(xz.shape[0] / max_points))]
    return xz


def _camera_project(
    points: np.ndarray,
    image_shape: tuple[int, int],
    camera_eye: tuple[float, float, float],
    camera_target: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    eye = np.asarray(camera_eye, dtype=np.float32)
    target = np.asarray(camera_target, dtype=np.float32)
    forward = target - eye
    forward = forward / max(np.linalg.norm(forward), 1e-6)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    rel = points.astype(np.float32) - eye
    cam_x = rel @ right
    cam_y = rel @ up
    cam_z = rel @ forward
    valid = cam_z > 0.02
    if not np.any(valid):
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    height, width = image_shape
    horizontal_fov = np.deg2rad(55.0)
    vertical_fov = 2.0 * np.arctan(np.tan(horizontal_fov / 2.0) * height / width)
    x_ndc = cam_x[valid] / (cam_z[valid] * np.tan(horizontal_fov / 2.0))
    y_ndc = cam_y[valid] / (cam_z[valid] * np.tan(vertical_fov / 2.0))
    u = ((x_ndc + 1.0) * 0.5 * width).astype(np.int32)
    v = ((1.0 - y_ndc) * 0.5 * height).astype(np.int32)
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(in_frame):
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    return u[in_frame], v[in_frame]


def _draw_points(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
    radius: int,
) -> None:
    if u.size == 0:
        return
    height, width = image.shape[:2]
    color_arr = np.array(color, dtype=np.float32)
    for x, y in zip(u, v):
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        patch = image[y0:y1, x0:x1].astype(np.float32)
        image[y0:y1, x0:x1] = (patch * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)


def _draw_target_overlay(
    image: np.ndarray,
    task_env,
    camera_eye: tuple[float, float, float],
    camera_target: tuple[float, float, float],
    prior_volume: np.ndarray | None = None,
) -> np.ndarray:
    target_points = _target_vertebra_world_points(task_env, max_points=700)
    rec_points = _reconstruction_world_points(task_env, max_points=700)
    prior_points = _estimated_prior_world_points(
        task_env,
        prior_volume,
        args_cli.prior_overlay_threshold,
        args_cli.prior_overlay_max_points,
    )

    out = image.copy()
    shape = out.shape[:2]
    target_u, target_v = _camera_project(target_points, shape, camera_eye, camera_target)
    rec_u, rec_v = _camera_project(rec_points, shape, camera_eye, camera_target)
    prior_u, prior_v = _camera_project(prior_points, shape, camera_eye, camera_target)

    _draw_points(out, prior_u, prior_v, color=(255, 86, 178), alpha=0.48, radius=2)
    _draw_points(out, target_u, target_v, color=(0, 155, 60), alpha=0.62, radius=1)
    _draw_points(out, rec_u, rec_v, color=(75, 145, 205), alpha=0.78, radius=2)
    return out
def _projection_rgb(reconstruction: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Render an x-z max projection as RGB."""
    rec_proj = reconstruction.max(axis=1).T > 0
    rgb = np.zeros((*rec_proj.shape, 3), dtype=np.float32)
    rgb[...] = np.array([0.05, 0.06, 0.07], dtype=np.float32)

    if target is not None:
        target_proj = target.max(axis=1).T > 0
        rgb[target_proj] = np.array([0.32, 0.32, 0.34], dtype=np.float32)

    rgb[rec_proj] = np.array([0.00, 0.78, 0.68], dtype=np.float32)
    return rgb


def _cmd_to_volume_xz(env, cmd_state: np.ndarray) -> tuple[float, float]:
    rec = env.surface_reconstructor
    label_res = float(rec.label_res)
    volume_res = float(rec.volume_res)
    corner = rec.human_rec_volume_corner[0].detach().cpu().numpy()
    x = (cmd_state[0] * label_res - corner[0]) / volume_res
    z = (cmd_state[1] * label_res - corner[2]) / volume_res
    return float(x), float(z)


def _cmd_traj_to_volume_xz(env, cmd_traj: list[np.ndarray]) -> np.ndarray:
    if not cmd_traj:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([_cmd_to_volume_xz(env, cmd) for cmd in cmd_traj], dtype=np.float32)


def _cmd_traj_in_us_volume(env, cmd_traj: list[np.ndarray]) -> np.ndarray:
    """Draw the command path in the probe-observation volume bounds."""
    rec = env.surface_reconstructor
    vol_traj = _cmd_traj_to_volume_xz(env, cmd_traj)
    if not vol_traj.size:
        return np.empty((0, 3), dtype=np.float32)
    limit = float(rec.volume_size[0].item()) - 1.0
    vol_traj = np.clip(vol_traj, 0.0, limit)
    y = np.full(vol_traj.shape[0], 6.0, dtype=np.float32)
    return np.column_stack([vol_traj[:, 0], y, vol_traj[:, 1]]).astype(np.float32)


def _make_frame(env, step: int, coverage: list[float], cmd_traj: list[np.ndarray]) -> np.ndarray:
    rec = env.surface_reconstructor
    human_volume = rec.human_rec_volume[0].detach().cpu().numpy()
    us_volume = rec.US_rec_volume[0].detach().cpu().numpy()
    target_volume = rec.upper_surface_volume_list[0].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 848 / 120), dpi=120)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_facecolor("#101215")

    for ax in axes.ravel():
        ax.set_facecolor("#101215")
        ax.tick_params(colors="#b7bec7", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    axes[0, 0].imshow(_projection_rgb(human_volume, target_volume), origin="lower")
    axes[0, 0].set_title("累计重建与目标", color="#e8edf2", fontsize=10, fontproperties=CHINESE_FONT)
    axes[0, 0].set_xlabel("体素 x", color="#b7bec7", fontproperties=CHINESE_FONT)
    axes[0, 0].set_ylabel("体素 z", color="#b7bec7", fontproperties=CHINESE_FONT)

    if cmd_traj:
        vol_traj = np.array([_cmd_to_volume_xz(env, cmd) for cmd in cmd_traj], dtype=np.float32)
        axes[0, 0].plot(vol_traj[:, 0], vol_traj[:, 1], color="#ffcf5a", linewidth=1.4)
        axes[0, 0].scatter(vol_traj[-1, 0], vol_traj[-1, 1], color="#ff5a5f", s=18)

    axes[0, 1].imshow(_projection_rgb(us_volume), origin="lower")
    axes[0, 1].set_title("探头坐标系重建观测", color="#e8edf2", fontsize=10, fontproperties=CHINESE_FONT)
    axes[0, 1].set_xlabel("超声体素 x", color="#b7bec7", fontproperties=CHINESE_FONT)
    axes[0, 1].set_ylabel("超声体素 z", color="#b7bec7", fontproperties=CHINESE_FONT)

    axes[1, 0].plot(coverage, color="#4cc9f0", linewidth=2.0)
    axes[1, 0].fill_between(np.arange(len(coverage)), coverage, color="#4cc9f0", alpha=0.18)
    axes[1, 0].set_ylim(0.0, max(0.05, min(1.0, max(coverage + [0.0]) * 1.2)))
    axes[1, 0].set_title("目标表面覆盖率", color="#e8edf2", fontsize=10, fontproperties=CHINESE_FONT)
    axes[1, 0].set_xlabel("控制步", color="#b7bec7", fontproperties=CHINESE_FONT)
    axes[1, 0].set_ylabel("覆盖率", color="#b7bec7", fontproperties=CHINESE_FONT)
    axes[1, 0].grid(color="#30363d", linewidth=0.6, alpha=0.7)

    if cmd_traj:
        traj = np.stack(cmd_traj, axis=0)
        axes[1, 1].plot(traj[:, 0], traj[:, 1], color="#ffcf5a", linewidth=1.6)
        axes[1, 1].scatter(traj[-1, 0], traj[-1, 1], color="#ff5a5f", s=20)
        axes[1, 1].set_aspect("equal", adjustable="datalim")
    axes[1, 1].set_title(f"人体 x-z 平面{TRAJECTORY_LABEL}", color="#e8edf2", fontsize=10, fontproperties=CHINESE_FONT)
    axes[1, 1].set_xlabel("x 指令", color="#b7bec7", fontproperties=CHINESE_FONT)
    axes[1, 1].set_ylabel("z 指令", color="#b7bec7", fontproperties=CHINESE_FONT)
    axes[1, 1].grid(color="#30363d", linewidth=0.6, alpha=0.7)

    cov_value = coverage[-1] if coverage else 0.0
    fig.suptitle(
        f"{TRAJECTORY_LABEL}重建 | 步骤 {step:04d} | 覆盖率 {cov_value:.4f}",
        color="#f2f5f8",
        fontsize=13,
        fontproperties=CHINESE_FONT,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def _resize_cover(
    image: np.ndarray,
    width: int,
    height: int,
    zoom: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    scale = max(width / image.shape[1], height / image.shape[0]) * max(zoom, 1.0)
    resized_w = int(round(image.shape[1] * scale))
    resized_h = int(round(image.shape[0] * scale))

    try:
        import cv2

        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        resized = np.asarray(Image.fromarray(image).resize((resized_w, resized_h)))

    extra_y = max(0, resized_h - height)
    extra_x = max(0, resized_w - width)
    x0 = int(round(extra_x * (0.5 + 0.5 * float(np.clip(offset[0], -1.0, 1.0)))))
    y0 = int(round(extra_y * (0.5 + 0.5 * float(np.clip(offset[1], -1.0, 1.0)))))
    return resized[y0 : y0 + height, x0 : x0 + width, :3]


def _enhance_contact_crop(image: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except Exception:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    frame = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2RGB)
    frame = cv2.addWeighted(frame, 0.72, image, 0.28, 0)
    blur = cv2.GaussianBlur(frame, (0, 0), 1.2)
    frame = cv2.addWeighted(frame, 1.35, blur, -0.35, 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask = (gray < 145).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 450]
    if not contours:
        return frame

    contour = max(contours, key=cv2.contourArea)
    outline_bgr = (0, 112, 255)
    contact_bgr = (32, 32, 255)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.drawContours(frame_bgr, [contour], -1, outline_bgr, 2, lineType=cv2.LINE_AA)

    pts = contour.reshape(-1, 2)
    bottom_y = int(np.percentile(pts[:, 1], 96))
    contact_pts = pts[pts[:, 1] >= bottom_y - 10]
    if contact_pts.shape[0] >= 2:
        x0, x1 = np.percentile(contact_pts[:, 0], [8, 92]).astype(int)
        y = int(np.median(contact_pts[:, 1]))
        cv2.line(frame_bgr, (x0, y), (x1, y), contact_bgr, 4, lineType=cv2.LINE_AA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _voxel_points(volume: np.ndarray, max_points: int = 12000) -> np.ndarray:
    points = np.argwhere(volume > 0)
    if points.shape[0] > max_points:
        stride = int(np.ceil(points.shape[0] / max_points))
        points = points[::stride]
    return points.astype(np.float32)


def _vertebra_points_in_us_volume(env) -> np.ndarray:
    rec = env.surface_reconstructor
    vertebra = torch.as_tensor(rec.target_vertebra_points[0], device=rec.device).float() * rec.label_res
    ee_to_human_pos, ee_to_human_quat = subtract_frame_transforms(
        env.US_ee_pose_w[0:1, 0:3],
        env.US_ee_pose_w[0:1, 3:7],
        env.world_to_human_pos[0:1],
        env.world_to_human_rot[0:1],
    )
    ver_in_us = transform_points(vertebra, ee_to_human_pos, ee_to_human_quat)
    if ver_in_us.dim() == 2:
        ver_in_us = ver_in_us.unsqueeze(0)
    ver_in_us = ver_in_us[0]
    ver_in_us[:, 2] = ver_in_us[:, 2] - rec.height_img - rec.add_height
    ver_in_us[:, 0] += rec.real_volume_size[0] / 2
    ver_in_us[:, 1] += rec.real_volume_size[1] / 2
    ver_in_us = ver_in_us / rec.volume_res
    points = ver_in_us.detach().cpu().numpy()
    if points.shape[0] > 10000:
        points = points[:: int(np.ceil(points.shape[0] / 10000))]
    return points


def _style_3d_axis(ax, limit: int = 40) -> None:
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_zlim(0, limit)
    ax.view_init(elev=22, azim=-58)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    ax.set_facecolor(SITE_BACKGROUND)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))


def _add_left_projection_inset(
    ax_scene,
    task_env,
    covered_volume: np.ndarray,
    uncovered_volume: np.ndarray,
    cmd_traj: list[np.ndarray],
    prior_volume: np.ndarray | None = None,
) -> None:
    inset = ax_scene.inset_axes([0.045, 0.56, 0.55, 0.40])
    inset.set_facecolor((1.0, 1.0, 1.0, 0.86))

    uncovered_points = _voxel_points(uncovered_volume, max_points=6500)
    covered_points = _voxel_points(covered_volume, max_points=6500)
    prior_points = _prior_xz_points(prior_volume, args_cli.prior_overlay_threshold)
    if uncovered_points.size:
        inset.scatter(
            uncovered_points[:, 0],
            uncovered_points[:, 2],
            c="#3e65b2",
            s=9,
            alpha=0.34,
            linewidths=0,
        )
    if prior_points.size:
        inset.scatter(
            prior_points[:, 0],
            prior_points[:, 1],
            c="#ff56b2",
            s=16,
            alpha=0.32,
            linewidths=0,
        )
    if covered_points.size:
        inset.scatter(
            covered_points[:, 0],
            covered_points[:, 2],
            c="#e1d80b",
            s=14,
            alpha=0.88,
            linewidths=0,
        )
    vol_traj = _cmd_traj_to_volume_xz(task_env, cmd_traj)
    if vol_traj.size:
        inset.plot(
            vol_traj[:, 0],
            vol_traj[:, 1],
            color="#202020",
            linewidth=3.0,
            solid_capstyle="round",
            zorder=5,
        )
        inset.scatter(
            vol_traj[-1, 0],
            vol_traj[-1, 1],
            c="#ff6b00",
            s=46,
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
        )
    volume_limit = float(task_env.surface_reconstructor.volume_size[0].item())
    inset.set_xlim(0, volume_limit)
    inset.set_ylim(-0.04 * volume_limit, 1.28 * volume_limit)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.text(
        0.5,
        0.96,
        "二维目标\n投影",
        transform=inset.transAxes,
        fontsize=36,
        color="black",
        fontproperties=CHINESE_FONT,
        ha="center",
        va="top",
        linespacing=0.9,
    )
    inset.text(
        0.04,
        0.05,
        "蓝=真实未见  黄=已见  粉=估计prior  黑=轨迹",
        transform=inset.transAxes,
        fontsize=16,
        color="black",
        fontproperties=CHINESE_FONT,
        va="bottom",
    )
    for spine in inset.spines.values():
        spine.set_color("#8aa0aa")
        spine.set_linewidth(1.2)


def _add_contact_inset(ax_scene, scene_frame: np.ndarray) -> None:
    inset = ax_scene.inset_axes([0.49, 0.045, 0.48, 0.39])
    contact_frame = _resize_cover(scene_frame, 560, 390, zoom=4.7, offset=(0.82, -0.18))
    contact_frame = _enhance_contact_crop(contact_frame)
    inset.imshow(contact_frame)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.text(
        0.04,
        0.08,
        "探头接触特写",
        transform=inset.transAxes,
        fontsize=22,
        color="black",
        fontproperties=CHINESE_FONT,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
    )
    for spine in inset.spines.values():
        spine.set_color("#56616a")
        spine.set_linewidth(1.8)


def _make_site_frame(
    env_wrapper,
    task_env,
    step: int,
    cmd_traj: list[np.ndarray],
    render_scene: bool,
    scene_zoom: float,
    scene_offset: tuple[float, float],
    patient_prior_predictor=None,
) -> np.ndarray:
    rec = task_env.surface_reconstructor
    human_volume = rec.human_rec_volume[0].detach().cpu().numpy()
    us_volume = rec.US_rec_volume[0].detach().cpu().numpy()
    target_volume = rec.upper_surface_volume_list[0].detach().cpu().numpy() > 0
    covered_volume = np.logical_and(target_volume, human_volume > 0)
    uncovered_volume = np.logical_and(target_volume, ~covered_volume)
    coverage = float(rec.get_converage_ratio()[0].detach().cpu())
    estimated_prior_volume = (
        _estimated_patient_prior_volume(task_env, patient_prior_predictor)
        if args_cli.visualize_patient_prior
        else None
    )

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=SITE_BACKGROUND)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 0.6],
        height_ratios=[1.0, 1.0],
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
        wspace=0.0,
        hspace=0.0,
    )

    ax_scene = fig.add_subplot(gs[:, 0])
    if render_scene:
        task_env.sim.render()
        scene_frame = env_wrapper.render()
        contact_source_frame = None
        if scene_frame is None or scene_frame.size == 0 or float(scene_frame.std()) < 1.0:
            task_env.sim.render()
            scene_frame = env_wrapper.render()
        if scene_frame is None or scene_frame.size == 0 or float(scene_frame.std()) < 1.0:
            scene_frame = np.full((1080, 960, 3), 245, dtype=np.uint8)
        else:
            rendered_scene_frame = scene_frame[..., :3]
            scene_with_overlay = _draw_target_overlay(
                rendered_scene_frame.copy(),
                task_env,
                tuple(args_cli.camera_eye),
                tuple(args_cli.camera_target),
                estimated_prior_volume,
            )
            contact_source_frame = _resize_cover(rendered_scene_frame, 960, 1080, scene_zoom, scene_offset)
            scene_frame = _resize_cover(scene_with_overlay, 960, 1080, scene_zoom, scene_offset)
        ax_scene.imshow(scene_frame)
        ax_scene.axis("off")
        _add_left_projection_inset(
            ax_scene,
            task_env,
            covered_volume,
            uncovered_volume,
            cmd_traj,
            estimated_prior_volume,
        )
        if contact_source_frame is not None:
            _add_contact_inset(ax_scene, contact_source_frame)
        ax_scene.text(
            0.045,
            0.14,
            "绿=真实\n粉=估计prior\n蓝=重建",
            transform=ax_scene.transAxes,
            fontsize=36,
            color="black",
            fontproperties=CHINESE_FONT,
            linespacing=1.08,
        )
    else:
        ax_scene.remove()
        ax_scene = fig.add_subplot(gs[:, 0], projection="3d")
        _style_3d_axis(ax_scene, int(rec.volume_size[0].item()))
        left_uncovered_points = _voxel_points(uncovered_volume, max_points=18000)
        left_covered_points = _voxel_points(covered_volume, max_points=18000)
        if left_uncovered_points.size:
            ax_scene.scatter(
                left_uncovered_points[:, 0],
                left_uncovered_points[:, 1],
                left_uncovered_points[:, 2],
                c="#3e65b2",
                s=4,
                alpha=0.28,
                linewidths=0,
            )
        if left_covered_points.size:
            ax_scene.scatter(
                left_covered_points[:, 0],
                left_covered_points[:, 1],
                left_covered_points[:, 2],
                c="#e1d80b",
                s=8,
                alpha=0.9,
                linewidths=0,
            )
        if cmd_traj:
            vol_traj = _cmd_traj_to_volume_xz(task_env, cmd_traj)
            y = np.full(vol_traj.shape[0], 8.0, dtype=np.float32)
            ax_scene.plot(vol_traj[:, 0], y, vol_traj[:, 1], color="#202020", linewidth=2.0)
            ax_scene.scatter(vol_traj[-1, 0], y[-1], vol_traj[-1, 1], c="#d62728", s=45)
        ax_scene.view_init(elev=25, azim=-65)
        ax_scene.text2D(
            0.04,
            0.95,
            TRAJECTORY_LABEL,
            transform=ax_scene.transAxes,
            fontsize=32,
            color="black",
            fontproperties=CHINESE_FONT,
        )

    ax_us = fig.add_subplot(gs[0, 1], projection="3d")
    _style_3d_axis(ax_us, int(rec.volume_size[0].item()))
    us_points = _voxel_points(us_volume, max_points=9000)
    if us_points.size:
        ax_us.scatter(
            us_points[:, 0],
            us_points[:, 1],
            us_points[:, 2],
            c=us_points[:, 2],
            cmap="Greens",
            s=3,
            alpha=0.8,
            linewidths=0,
        )
    red_points = _vertebra_points_in_us_volume(task_env)
    ax_us.scatter(red_points[:, 0], red_points[:, 1], red_points[:, 2], c="#d62728", s=0.45, alpha=0.45, linewidths=0)
    ax_us.text2D(
        0.03,
        0.90,
        "探头坐标系观测",
        transform=ax_us.transAxes,
        fontsize=36,
        color="black",
        fontproperties=CHINESE_FONT,
    )
    legend_text = "红=真实\n绿=观测"
    fig.text(
        0.965,
        0.93,
        legend_text,
        ha="right",
        va="top",
        fontsize=30,
        color="black",
        fontproperties=CHINESE_FONT,
        linespacing=1.08,
    )
    ax_status = fig.add_subplot(gs[1, 1], projection="3d")
    _style_3d_axis(ax_status, int(rec.volume_size[0].item()))
    uncovered_points = _voxel_points(uncovered_volume, max_points=12000)
    covered_points = _voxel_points(covered_volume, max_points=12000)
    prior_points = _prior_volume_points(
        estimated_prior_volume,
        args_cli.prior_overlay_threshold,
        max_points=6000,
    )
    if uncovered_points.size:
        ax_status.scatter(
            uncovered_points[:, 0],
            uncovered_points[:, 1],
            uncovered_points[:, 2],
            c="#3e65b2",
            s=3,
            alpha=0.42,
            linewidths=0,
        )
    if prior_points.size:
        ax_status.scatter(
            prior_points[:, 0],
            prior_points[:, 1],
            prior_points[:, 2],
            c="#ff56b2",
            s=4,
            alpha=0.32,
            linewidths=0,
        )
    if covered_points.size:
        ax_status.scatter(
            covered_points[:, 0],
            covered_points[:, 1],
            covered_points[:, 2],
            c="#e1d80b",
            s=5,
            alpha=0.9,
            linewidths=0,
        )
    status_traj = _cmd_traj_in_us_volume(task_env, cmd_traj)
    if status_traj.size:
        ax_status.plot(
            status_traj[:, 0],
            status_traj[:, 1],
            status_traj[:, 2],
            color="#202020",
            linewidth=2.4,
            alpha=0.95,
        )
        ax_status.scatter(
            status_traj[-1, 0],
            status_traj[-1, 1],
            status_traj[-1, 2],
            c="#ff6b00",
            s=46,
            edgecolors="white",
            linewidths=0.8,
        )
    ax_status.text2D(
        0.03,
        0.96,
        "重建覆盖状态\n蓝=真实未建\n黄=已建\n粉=估计prior\n黑=轨迹",
        transform=ax_status.transAxes,
        fontsize=16,
        color="black",
        fontproperties=CHINESE_FONT,
        linespacing=1.12,
    )
    ax_status.text2D(
        0.50,
        0.06,
        f"覆盖率 {coverage:.3f}",
        transform=ax_status.transAxes,
        ha="left",
        va="bottom",
        fontsize=38,
        color="black",
        fontproperties=CHINESE_FONT,
        linespacing=1.04,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3},
    )

    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.layout == "site" and args_cli.render_scene:
        env_cfg.viewer.resolution = (960, 1080)
    render_mode = "rgb_array" if args_cli.layout == "site" and args_cli.render_scene else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    task_env = env.unwrapped
    if args_cli.layout == "site" and args_cli.render_scene:
        task_env.sim.set_camera_view(eye=list(args_cli.camera_eye), target=list(args_cli.camera_target))

    human_pos_2d_min = task_env.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]] - args_cli.square_size
    human_pos_2d_max = task_env.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]] + args_cli.square_size

    action_helper = HeuristicReconstruction(
        max_action=task_env.max_action,
        action_scale=task_env.action_scale,
        human_pos_2d_min=human_pos_2d_min,
        human_pos_2d_max=human_pos_2d_max,
        num_sections=2,
        total_steps=args_cli.planner_total_steps,
        ratio=[0.05, 0.05, 0.05, 0.0],
        device=task_env.sim.device,
    )
    nbv_planner = None
    patient_prior_predictor = None
    policy_components = None
    saved_action_components = None
    if args_cli.trajectory_source == "nbv":
        anatomy_prior = load_prior(args_cli.anatomy_prior, task_env.sim.device)
        nbv_planner = OnlineNBVGoalPlanner(
            task_env,
            anatomy_prior["prior_volume"].float(),
            args_cli.nbv_replan_interval,
            args_cli.nbv_reach_radius,
            args_cli.nbv_observation_radius,
            args_cli.nbv_distance_weight,
            args_cli.nbv_visit_weight,
            args_cli.nbv_yaw,
        )
    elif args_cli.trajectory_source == "aus_slam":
        anatomy_prior = load_prior(args_cli.anatomy_prior, task_env.sim.device)
        patient_prior_predictor = (
            load_patient_prior_predictor(args_cli.patient_prior_model, task_env.sim.device)
            if args_cli.patient_prior_model
            else None
        )
        if args_cli.registration_prior:
            patient_prior_predictor = RegistrationPriorPredictor(anatomy_prior["prior_volume"].float())
        view_gain_predictor = (
            load_view_gain_predictor(args_cli.view_gain_model, task_env.sim.device)
            if args_cli.view_gain_model
            else None
        )
        nbv_planner = ActiveUSSLAMGoalPlanner(
            task_env,
            anatomy_prior["prior_volume"].float(),
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
    elif args_cli.trajectory_source == "aus_rh_slam":
        anatomy_prior = load_prior(args_cli.anatomy_prior, task_env.sim.device)
        patient_prior_predictor = (
            load_patient_prior_predictor(args_cli.patient_prior_model, task_env.sim.device)
            if args_cli.patient_prior_model
            else None
        )
        if args_cli.registration_prior:
            patient_prior_predictor = RegistrationPriorPredictor(anatomy_prior["prior_volume"].float())
        view_gain_predictor = (
            load_view_gain_predictor(args_cli.view_gain_model, task_env.sim.device)
            if args_cli.view_gain_model
            else None
        )
        nbv_planner = RecedingHorizonActiveUSSLAMGoalPlanner(
            task_env,
            anatomy_prior["prior_volume"].float(),
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
    elif args_cli.trajectory_source == "policy":
        policy_components = _load_policy_components(task_env, args_cli.trajectory)
        nbv_planner = policy_components[-1]
    elif args_cli.trajectory_source in ("stitching", "random", "expert_replay", "aus_rh_replay"):
        saved_action_components = _load_saved_actions(
            args_cli.trajectory,
            args_cli.trajectory_index,
            args_cli.trajectory_source,
        )

    output_dir = Path(args_cli.output_dir)
    thumbnail_dir = Path(args_cli.thumbnail_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix_by_source = {
        "expert": "expert_reconstruction",
        "nbv": "nbv_reconstruction",
        "policy": "policy_reconstruction",
        "stitching": "diffstitch_reconstruction",
        "random": "random_reconstruction",
        "expert_replay": "expert_replay_reconstruction",
        "aus_slam": "aus_slam_reconstruction",
        "aus_rh_slam": "aus_rh_slam_reconstruction",
        "aus_rh_replay": "aus_rh_slam_replay_reconstruction",
    }
    patient_suffix = f"_{resolved_patient_ids[0]}" if len(resolved_patient_ids) == 1 else ""
    prefix = f"{prefix_by_source[args_cli.trajectory_source]}{patient_suffix}"
    video_path = output_dir / f"{prefix}_{stamp}.mp4"
    thumbnail_path = thumbnail_dir / f"{prefix}_{stamp}.png"

    obs, info = env.reset()
    del obs
    if nbv_planner is not None:
        nbv_planner.reset()
    if args_cli.layout == "site" and args_cli.render_scene:
        _set_scene_render_style(task_env)
        _set_target_anatomy_scene_overlay(task_env)

    coverage: list[float] = []
    cmd_traj: list[np.ndarray] = []
    frames_written = 0
    total_steps = int(args_cli.steps)
    if saved_action_components is not None:
        _, saved_actions = saved_action_components
        total_steps = min(total_steps, int(saved_actions.shape[0]))

    with imageio.get_writer(video_path, fps=args_cli.fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for step in range(total_steps):
            cmd_traj.append(info["cur_cmd_state"][0].detach().cpu().numpy())
            coverage.append(float(task_env.surface_reconstructor.get_converage_ratio()[0].detach().cpu()))

            if step % args_cli.capture_interval == 0:
                if args_cli.layout == "site":
                    frame = _make_site_frame(
                        env,
                        task_env,
                        step,
                        cmd_traj,
                        args_cli.render_scene,
                        args_cli.scene_zoom,
                        tuple(args_cli.scene_offset),
                        patient_prior_predictor,
                    )
                else:
                    frame = _make_frame(task_env, step, coverage, cmd_traj)
                writer.append_data(frame)
                frames_written += 1

            with torch.inference_mode():
                if args_cli.trajectory_source == "expert":
                    actions = action_helper.get_action(info, step)
                elif args_cli.trajectory_source in ("nbv", "aus_slam", "aus_rh_slam"):
                    goal_cmd_pose = nbv_planner.goal(info["cur_cmd_state"], step).to(task_env.sim.device)
                    actions = action_helper.get_action_given_goal(info, goal_cmd_pose)
                elif args_cli.trajectory_source == "policy":
                    (
                        _data,
                        state_mode,
                        actor,
                        action_norm,
                        cmd_norm,
                        action_min,
                        action_max,
                        us_slicer,
                        policy_goal_planner,
                    ) = policy_components
                    live_image = _live_policy_image(task_env, us_slicer)
                    live_cmd_raw = _live_policy_command_state(info, state_mode, policy_goal_planner, step)
                    live_cmd = cmd_norm.encode(live_cmd_raw).to(task_env.sim.device)
                    action_encoded = actor(live_image, live_cmd, deterministic=True)
                    actions = action_norm.decode(action_encoded.cpu()).to(task_env.sim.device)
                    if action_min is not None and action_max is not None:
                        actions = torch.clamp(actions, min=action_min, max=action_max)
                else:
                    _data, saved_actions = saved_action_components
                    actions = saved_actions[step].to(task_env.sim.device).view(1, -1)
                    if task_env.scene.num_envs > 1:
                        actions = actions.repeat(task_env.scene.num_envs, 1)
                _, _, terminated, truncated, info = env.step(actions)

            if torch.any(torch.logical_or(terminated, truncated)).item():
                break

        if args_cli.layout == "site":
            final_frame = _make_site_frame(
                env,
                task_env,
                step,
                cmd_traj,
                args_cli.render_scene,
                args_cli.scene_zoom,
                tuple(args_cli.scene_offset),
                patient_prior_predictor,
            )
        else:
            final_frame = _make_frame(task_env, step, coverage, cmd_traj)
        writer.append_data(final_frame)
        frames_written += 1

    imageio.imwrite(thumbnail_path, final_frame)
    final_coverage = coverage[-1] if coverage else 0.0
    print(f"[RESULT] video={video_path.resolve()}")
    print(f"[RESULT] thumbnail={thumbnail_path.resolve()}")
    print(f"[RESULT] frames={frames_written}")
    print(f"[RESULT] steps={len(coverage)}")
    print(f"[RESULT] final_coverage={final_coverage:.6f}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

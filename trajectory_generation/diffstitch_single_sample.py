"""Generate one DiffStitch-style offline RL training sample.

This is a lightweight prototype of trajectory stitching for the retained
SonoGym offline-RL dataset.  It trains a small conditional DDPM on short
sub-trajectories, samples one bridge between a low-return prefix and a
high-return suffix, fills bridge ultrasound frames by nearest-neighbor lookup
in command-state space, and saves the resulting single trajectory as a normal
`.pt` training sample.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


parser = argparse.ArgumentParser(description="Create one DiffStitch-style stitched trajectory sample.")
parser.add_argument("--input", type=str, default="artifacts/trajectories/random_16_us_prior_train.pt")
parser.add_argument("--output", type=str, default="artifacts/trajectories/diffstitch_single_sample.pt")
parser.add_argument("--bridge_length", type=int, default=32)
parser.add_argument("--prefix_steps", type=int, default=64)
parser.add_argument("--suffix_steps", type=int, default=96)
parser.add_argument("--iterations", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--diffusion_steps", type=int, default=80)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--reward_key", choices=("prior_gain_reward", "proxy_reward"), default="prior_gain_reward")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
args = parser.parse_args()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(math.log(10000.0) / max(half - 1, 1))
        )
        args_t = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args_t), torch.cos(args_t)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class BridgeDenoiser(nn.Module):
    def __init__(self, x_dim: int, cond_dim: int, hidden_dim: int):
        super().__init__()
        time_dim = 64
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(x_dim + cond_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, noisy_x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([noisy_x, cond, self.time(t)], dim=-1))


def _device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_features(data: dict, reward_key: str) -> tuple[torch.Tensor, list[str]]:
    reward = data[reward_key].float().unsqueeze(-1)
    parts = [
        data["cmd_state"].float(),
        data["goal_cmd_pose"].float(),
        data["action"].float(),
        reward,
    ]
    names = ["cmd_state", "goal_cmd_pose", "action", reward_key]
    return torch.cat(parts, dim=-1), names


def _window_dataset(
    features: torch.Tensor,
    cmd_state: torch.Tensor,
    goal: torch.Tensor,
    reward: torch.Tensor,
    bridge_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_traj, trajectory_length, _ = features.shape
    if trajectory_length < bridge_length + 2:
        raise ValueError(f"trajectory_length={trajectory_length} is too short for bridge_length={bridge_length}.")

    windows = []
    conds = []
    returns = []
    for traj_idx in range(num_traj):
        for start in range(0, trajectory_length - bridge_length + 1):
            end = start + bridge_length - 1
            window_return = reward[traj_idx, start : end + 1].sum()
            cond = torch.cat(
                [
                    cmd_state[traj_idx, start],
                    cmd_state[traj_idx, end],
                    goal[traj_idx, end],
                    window_return.reshape(1),
                ],
                dim=0,
            )
            windows.append(features[traj_idx, start : end + 1].reshape(-1))
            conds.append(cond)
            returns.append(window_return)
    return torch.stack(windows), torch.stack(conds), torch.stack(returns)


def _normalizer(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return value.mean(0), value.std(0).clamp_min(1e-6)


def _ddpm_schedule(num_steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def _train_diffusion(
    windows: torch.Tensor,
    conds: torch.Tensor,
    device: torch.device,
) -> tuple[BridgeDenoiser, dict, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    x_mean, x_std = _normalizer(windows)
    c_mean, c_std = _normalizer(conds)
    x_train = (windows - x_mean) / x_std
    c_train = (conds - c_mean) / c_std

    model = BridgeDenoiser(x_train.shape[-1], c_train.shape[-1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    betas, alphas, alpha_bars = _ddpm_schedule(args.diffusion_steps, device)

    num_samples = x_train.shape[0]
    last_loss = torch.tensor(0.0)
    for iteration in range(args.iterations):
        idx = torch.randint(0, num_samples, (args.batch_size,))
        x0 = x_train[idx].to(device)
        cond = c_train[idx].to(device)
        t = torch.randint(0, args.diffusion_steps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        alpha_bar = alpha_bars[t].unsqueeze(-1)
        noisy = alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise
        pred = model(noisy, cond, t)
        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = loss.detach().cpu()
        if (iteration + 1) % max(args.iterations // 5, 1) == 0:
            print(f"[TRAIN] iteration={iteration + 1}/{args.iterations} loss={last_loss.item():.6f}")

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "cond_mean": c_mean,
        "cond_std": c_std,
        "last_loss": float(last_loss.item()),
    }
    return model, stats, (betas, alphas, alpha_bars)


@torch.no_grad()
def _sample_bridge(
    model: BridgeDenoiser,
    cond: torch.Tensor,
    stats: dict,
    schedule: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    betas, alphas, alpha_bars = schedule
    x_dim = stats["x_mean"].numel()
    cond_norm = ((cond - stats["cond_mean"]) / stats["cond_std"]).unsqueeze(0).to(device)
    x = torch.randn((1, x_dim), device=device)
    for step in reversed(range(args.diffusion_steps)):
        t = torch.full((1,), step, device=device, dtype=torch.long)
        pred_noise = model(x, cond_norm, t)
        alpha = alphas[step]
        alpha_bar = alpha_bars[step]
        beta = betas[step]
        x = (x - ((1.0 - alpha) / (1.0 - alpha_bar).sqrt()) * pred_noise) / alpha.sqrt()
        if step > 0:
            x = x + beta.sqrt() * torch.randn_like(x)
    return (x.cpu().squeeze(0) * stats["x_std"] + stats["x_mean"]).reshape(args.bridge_length, -1)


def _choose_stitch_pair(data: dict, reward_key: str) -> dict:
    rewards = data[reward_key].float()
    final_coverage = data["final_coverage"].float()
    high_idx = int(torch.argmax(final_coverage).item())
    low_idx = int(torch.argmin(final_coverage).item())
    trajectory_length = rewards.shape[1]
    bridge_length = args.bridge_length

    prefix_end = min(max(args.prefix_steps - 1, 1), trajectory_length - bridge_length - 2)
    suffix_start_min = bridge_length
    suffix_start_max = max(suffix_start_min, trajectory_length - args.suffix_steps - 1)
    suffix_candidates = torch.arange(suffix_start_min, suffix_start_max + 1)

    cmd = data["cmd_state"].float()
    cmd_scale = cmd.reshape(-1, cmd.shape[-1]).std(0).clamp_min(1e-6)
    start = cmd[low_idx, prefix_end]
    high_cmd = cmd[high_idx, suffix_candidates]
    dist = ((high_cmd - start) / cmd_scale).pow(2).sum(-1).sqrt()
    best_pos = int(torch.argmin(dist).item())
    suffix_start = int(suffix_candidates[best_pos].item())
    return {
        "low_idx": low_idx,
        "high_idx": high_idx,
        "prefix_end": prefix_end,
        "suffix_start": suffix_start,
        "normalized_cmd_distance": float(dist[best_pos].item()),
    }


def _nearest_source_indices(data: dict, bridge_cmd: torch.Tensor) -> torch.Tensor:
    source_cmd = data["cmd_state"].float().reshape(-1, data["cmd_state"].shape[-1])
    scale = source_cmd.std(0).clamp_min(1e-6)
    dist = torch.cdist(bridge_cmd.float() / scale, source_cmd / scale)
    return torch.argmin(dist, dim=1)


def _nearest_source_images(data: dict, bridge_cmd: torch.Tensor) -> torch.Tensor:
    if "us_image" not in data:
        raise KeyError("Input data must contain `us_image` to save a training sample.")
    source_images = data["us_image"].reshape(-1, *data["us_image"].shape[2:])
    nearest = _nearest_source_indices(data, bridge_cmd)
    return source_images[nearest]


def _trajectory_patient_id(data: dict, index: int) -> str:
    patient_ids = data.get("trajectory_patient_ids")
    if isinstance(patient_ids, list) and 0 <= index < len(patient_ids):
        return str(patient_ids[index])
    return f"traj{index}"


def _assemble_sample(data: dict, bridge: torch.Tensor, pair: dict) -> dict:
    cmd_dim = data["cmd_state"].shape[-1]
    goal_dim = data["goal_cmd_pose"].shape[-1]
    action_dim = data["action"].shape[-1]
    reward_offset = cmd_dim + goal_dim + action_dim

    low_idx = pair["low_idx"]
    high_idx = pair["high_idx"]
    prefix_end = pair["prefix_end"]
    suffix_start = pair["suffix_start"]
    suffix_end = min(suffix_start + args.suffix_steps, data["cmd_state"].shape[1])

    bridge_cmd = bridge[:, :cmd_dim].float()
    bridge_goal = bridge[:, cmd_dim : cmd_dim + goal_dim].float()
    bridge_action = bridge[:, cmd_dim + goal_dim : reward_offset].float()
    bridge_reward = bridge[:, reward_offset].float().clamp_min(0.0)

    action_min = data["action"].float().reshape(-1, action_dim).min(0).values
    action_max = data["action"].float().reshape(-1, action_dim).max(0).values
    bridge_action = torch.clamp(bridge_action, min=action_min, max=action_max)

    source_reward = data[args.reward_key].float()
    cmd = torch.cat(
        [
            data["cmd_state"][low_idx, : prefix_end + 1].float(),
            bridge_cmd,
            data["cmd_state"][high_idx, suffix_start:suffix_end].float(),
        ],
        dim=0,
    )
    goal = torch.cat(
        [
            data["goal_cmd_pose"][low_idx, : prefix_end + 1].float(),
            bridge_goal,
            data["goal_cmd_pose"][high_idx, suffix_start:suffix_end].float(),
        ],
        dim=0,
    )
    action = torch.cat(
        [
            data["action"][low_idx, : prefix_end + 1].float(),
            bridge_action,
            data["action"][high_idx, suffix_start:suffix_end].float(),
        ],
        dim=0,
    )
    reward = torch.cat(
        [
            source_reward[low_idx, : prefix_end + 1],
            bridge_reward,
            source_reward[high_idx, suffix_start:suffix_end],
        ],
        dim=0,
    )

    bridge_nearest = _nearest_source_indices(data, bridge_cmd)
    bridge_images = data["us_image"].reshape(-1, *data["us_image"].shape[2:])[bridge_nearest]
    us_image = torch.cat(
        [
            data["us_image"][low_idx, : prefix_end + 1],
            bridge_images,
            data["us_image"][high_idx, suffix_start:suffix_end],
        ],
        dim=0,
    )

    trajectory_length = cmd.shape[0]
    prefix_mask = torch.zeros(trajectory_length, dtype=torch.bool)
    bridge_mask = torch.zeros(trajectory_length, dtype=torch.bool)
    suffix_mask = torch.zeros(trajectory_length, dtype=torch.bool)
    prefix_mask[: prefix_end + 1] = True
    bridge_mask[prefix_end + 1 : prefix_end + 1 + args.bridge_length] = True
    suffix_mask[prefix_end + 1 + args.bridge_length :] = True
    segment_id = torch.zeros(trajectory_length, dtype=torch.long)
    segment_id[bridge_mask] = 1
    segment_id[suffix_mask] = 2

    proxy_reward = reward.clone()
    reconstructed_volume_voxels = torch.cumsum(proxy_reward, dim=0)
    delta_reconstructed_volume_voxels = proxy_reward

    output = {
        "task": data.get("task", "Isaac-robot-US-reconstruction-v0"),
        "kind": "diffstitch_single_training_sample",
        "source_file": args.input,
        "split": data.get("split"),
        "split_file": data.get("split_file"),
        "patient_ids": data.get("patient_ids"),
        "trajectory_patient_ids": [
            f"diffstitch:{_trajectory_patient_id(data, low_idx)}+{_trajectory_patient_id(data, high_idx)}"
        ],
        "num_traj": 1,
        "trajectory_length": int(trajectory_length),
        "expert_trajectory_length": int(trajectory_length),
        "cmd_state": cmd.unsqueeze(0),
        "goal_cmd_pose": goal.unsqueeze(0),
        "action": action.unsqueeze(0),
        "us_image": us_image.unsqueeze(0).to(torch.uint8),
        args.reward_key: reward.unsqueeze(0),
        "proxy_reward": proxy_reward.unsqueeze(0),
        "reconstructed_volume_voxels": reconstructed_volume_voxels.unsqueeze(0),
        "delta_reconstructed_volume_voxels": delta_reconstructed_volume_voxels.unsqueeze(0),
        "reconstructed_volume": reconstructed_volume_voxels.unsqueeze(0),
        "delta_reconstructed_volume": delta_reconstructed_volume_voxels.unsqueeze(0),
        "final_coverage": data["final_coverage"][high_idx : high_idx + 1].float(),
        "mean_final_coverage": float(data["final_coverage"][high_idx].float().item()),
        "stitched_mask": bridge_mask.unsqueeze(0),
        "source_segment_id": segment_id.unsqueeze(0),
        "diffstitch": {
            **pair,
            "bridge_length": int(args.bridge_length),
            "prefix_steps": int(prefix_end + 1),
            "suffix_steps": int(suffix_end - suffix_start),
            "reward_key": args.reward_key,
            "image_fill": "nearest source us_image in normalized cmd_state space",
        },
        "params": {
            **data.get("params", {}),
            "diffstitch_iterations": int(args.iterations),
            "diffusion_steps": int(args.diffusion_steps),
            "diffstitch_hidden_dim": int(args.hidden_dim),
        },
    }

    for key in ("anatomy_prior", "anatomy_prior_volume", "human_pos_2d_min", "human_pos_2d_max"):
        if key in data:
            output[key] = data[key]
    if "us_ct_image" in data:
        bridge_ct = data["us_ct_image"].reshape(-1, *data["us_ct_image"].shape[2:])[bridge_nearest]
        output["us_ct_image"] = torch.cat(
            [
                data["us_ct_image"][low_idx, : prefix_end + 1],
                bridge_ct,
                data["us_ct_image"][high_idx, suffix_start:suffix_end],
            ],
            dim=0,
        ).unsqueeze(0)
    return output


def main() -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device_from_arg(args.device)
    data = torch.load(args.input, map_location="cpu")
    for key in ("cmd_state", "goal_cmd_pose", "action", args.reward_key, "final_coverage", "us_image"):
        if key not in data:
            raise KeyError(f"Input trajectory file is missing `{key}`.")

    features, feature_names = _make_features(data, args.reward_key)
    windows, conds, window_returns = _window_dataset(
        features,
        data["cmd_state"].float(),
        data["goal_cmd_pose"].float(),
        data[args.reward_key].float(),
        args.bridge_length,
    )
    print(f"[DATA] windows={tuple(windows.shape)} feature_names={feature_names}")
    print(f"[DATA] window_return_mean={window_returns.mean().item():.6f}")

    model, stats, schedule = _train_diffusion(windows, conds, device)
    pair = _choose_stitch_pair(data, args.reward_key)
    cond_return = window_returns.quantile(0.85).reshape(1)
    cond = torch.cat(
        [
            data["cmd_state"][pair["low_idx"], pair["prefix_end"]].float(),
            data["cmd_state"][pair["high_idx"], pair["suffix_start"]].float(),
            data["goal_cmd_pose"][pair["high_idx"], pair["suffix_start"]].float(),
            cond_return,
        ],
        dim=0,
    )
    bridge = _sample_bridge(model, cond, stats, schedule, device)
    output = _assemble_sample(data, bridge, pair)

    output["diffstitch"]["training_window_return_q85"] = float(cond_return.item())
    output["diffstitch"]["denoising_loss"] = stats["last_loss"]
    output["diffstitch"]["model"] = "conditional_ddpm_mlp_low_dim_bridge"

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(output, args.output)
    print(f"[RESULT] saved one DiffStitch sample to {args.output}")
    print(f"[RESULT] trajectory_length={output['trajectory_length']}")
    print(f"[RESULT] bridge_steps={int(output['stitched_mask'].sum().item())}")
    print(
        "[RESULT] low_idx={low_idx} high_idx={high_idx} prefix_end={prefix_end} "
        "suffix_start={suffix_start} cmd_distance={normalized_cmd_distance:.4f}".format(**pair)
    )
    print(f"[RESULT] us_image_shape={tuple(output['us_image'].shape)}")


if __name__ == "__main__":
    main()

"""Train an offline RL policy from saved reconstruction trajectories.

The learned policy is pi(a | s), where s is a saved SonoGym image observation
and a is the 4D reconstruction action.
"""

from __future__ import annotations

import argparse
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

from trajectory_generation.image_offline_policy import (
    ImageCommandActor,
    ImageCommandCritic,
    Normalizer,
)


parser = argparse.ArgumentParser(description="Train an offline RL policy pi(a|s) from saved image observations.")
parser.add_argument("--input", type=str, default="artifacts/trajectories/random_16_us_prior_train.pt")
parser.add_argument("--output", type=str, default="artifacts/trajectories/offline_rl_optimized_trajectory.pt")
parser.add_argument("--iterations", type=int, default=2500)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--bc_weight", type=float, default=2.5)
parser.add_argument("--alpha", type=float, default=0.02, help="Entropy weight for stochastic pi(a|s).")
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--policy_delay", type=int, default=2)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--feature_dim", type=int, default=128)
parser.add_argument("--algorithm", choices=("sac_bc",), default="sac_bc")
parser.add_argument(
    "--state_mode",
    choices=(
        "us_image_goal_cmd",
        "us_image_cmd",
    ),
    default="us_image_goal_cmd",
)
parser.add_argument(
    "--reward_key",
    choices=("prior_gain_reward", "proxy_reward"),
    default="prior_gain_reward",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
args = parser.parse_args()


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def make_step_rewards(
    data: dict,
    final_coverage: torch.Tensor,
    num_steps: int,
    reward_key: str,
) -> tuple[torch.Tensor, str]:
    if reward_key not in data or data[reward_key] is None:
        raise KeyError(f"Requested reward key `{reward_key}` is not available in the trajectory file.")
    rewards = data[reward_key].float()
    if rewards.ndim != 2:
        raise ValueError(f"`{reward_key}` must have shape (num_traj, trajectory_length).")
    if rewards.shape[1] < num_steps:
        raise ValueError(f"`{reward_key}` is shorter than the available transition sequence.")
    descriptions = {
        "prior_gain_reward": "anatomy-prior probability gained by new reconstruction",
        "proxy_reward": "target-structure volume gained by new reconstruction",
    }
    return rewards[:, :num_steps], f"{reward_key}: {descriptions[reward_key]}"


def actor_sample(actor: nn.Module, image: torch.Tensor, command: torch.Tensor | None):
    return actor.sample(image, command)


def actor_log_prob(actor: nn.Module, image: torch.Tensor, command: torch.Tensor | None, action: torch.Tensor):
    return actor.log_prob(image, command, action)


def actor_forward(actor: nn.Module, image: torch.Tensor, command: torch.Tensor | None, deterministic: bool):
    return actor(image, command, deterministic=deterministic)


def critic_forward(critic: nn.Module, image: torch.Tensor, command: torch.Tensor | None, action: torch.Tensor):
    return critic(image, command, action)


def image_key_from_state_mode(state_mode: str) -> str:
    if state_mode in ("us_image_goal_cmd", "us_image_cmd"):
        return "us_image"
    raise ValueError(f"Unsupported state_mode `{state_mode}`.")


def uses_command_state(state_mode: str) -> bool:
    return state_mode in ("us_image_goal_cmd", "us_image_cmd")


def uses_goal_command_state(state_mode: str) -> bool:
    return state_mode == "us_image_goal_cmd"


def state_image_from_data(data: dict, state_mode: str) -> tuple[torch.Tensor, str, int]:
    image_key = image_key_from_state_mode(state_mode)
    if image_key not in data:
        raise KeyError(f"Input trajectory file must contain `{image_key}`. Regenerate training trajectories first.")
    return data[image_key].to(torch.uint8), image_key, 1


def trainable_parameters(module: nn.Module):
    return (param for param in module.parameters() if param.requires_grad)


def command_state_from_data(
    cmd_state: torch.Tensor,
    goal_cmd_pose: torch.Tensor | None,
    state_mode: str,
) -> torch.Tensor:
    if not uses_goal_command_state(state_mode):
        return cmd_state
    if goal_cmd_pose is None:
        raise KeyError(f"`{state_mode}` requires `goal_cmd_pose` in the trajectory file.")
    if goal_cmd_pose.shape[:2] != cmd_state.shape[:2]:
        raise ValueError(
            f"`goal_cmd_pose` shape {tuple(goal_cmd_pose.shape)} is incompatible with "
            f"`cmd_state` shape {tuple(cmd_state.shape)}."
        )
    goal_dim = goal_cmd_pose.shape[-1]
    goal_delta = goal_cmd_pose.float() - cmd_state[..., :goal_dim].float()
    return torch.cat([cmd_state.float(), goal_cmd_pose.float(), goal_delta], dim=-1)


def main() -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data = torch.load(args.input, map_location="cpu")
    state_image, image_key, input_channels = state_image_from_data(data, args.state_mode)
    cmd_state = data["cmd_state"].float()
    goal_cmd_pose = data.get("goal_cmd_pose")
    if goal_cmd_pose is not None:
        goal_cmd_pose = goal_cmd_pose.float()
    action = data["action"].float()
    final_coverage = data["final_coverage"].float()
    command_state = command_state_from_data(cmd_state, goal_cmd_pose, args.state_mode)
    use_command = uses_command_state(args.state_mode)

    num_traj, trajectory_length = state_image.shape[:2]
    action_dim = action.shape[-1]
    if trajectory_length < 2:
        raise ValueError("trajectory_length must be at least 2 for transition training.")

    states = state_image[:, :-1].reshape(-1, *state_image.shape[2:])
    next_states = state_image[:, 1:].reshape(-1, *state_image.shape[2:])
    cmd_states_raw = command_state[:, :-1, :].reshape(-1, command_state.shape[-1])
    next_cmd_states_raw = command_state[:, 1:, :].reshape(-1, command_state.shape[-1])
    actions_raw = action[:, :-1, :].reshape(-1, action_dim)
    rewards_2d, reward_description = make_step_rewards(data, final_coverage, trajectory_length - 1, args.reward_key)
    rewards = rewards_2d.reshape(-1)
    dones = torch.zeros((num_traj, trajectory_length - 1), dtype=torch.float32)
    dones[:, -1] = 1.0
    dones = dones.reshape(-1)

    action_norm = Normalizer(actions_raw.mean(0), actions_raw.std(0).clamp_min(1e-6))
    actions = action_norm.encode(actions_raw)
    command_states_raw = cmd_states_raw
    next_command_states_raw = next_cmd_states_raw
    cmd_norm = Normalizer(command_states_raw.mean(0), command_states_raw.std(0).clamp_min(1e-6))
    cmd_states = cmd_norm.encode(command_states_raw)
    next_cmd_states = cmd_norm.encode(next_command_states_raw)
    action_limit = actions.abs().amax(dim=0).clamp_min(1.0).to(device)

    command_dim = cmd_states.shape[-1]
    actor = ImageCommandActor(
        action_dim, command_dim, args.hidden_dim, args.feature_dim, action_limit, input_channels
    ).to(device)
    actor_target = ImageCommandActor(
        action_dim, command_dim, args.hidden_dim, args.feature_dim, action_limit, input_channels
    ).to(device)
    critic = ImageCommandCritic(action_dim, command_dim, args.hidden_dim, args.feature_dim, input_channels).to(device)
    critic_target = ImageCommandCritic(
        action_dim, command_dim, args.hidden_dim, args.feature_dim, input_channels
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.Adam(trainable_parameters(actor), lr=3e-4)
    critic_optimizer = torch.optim.Adam(trainable_parameters(critic), lr=3e-4)

    num_samples = states.shape[0]
    last_actor_loss = torch.tensor(0.0)
    last_critic_loss = torch.tensor(0.0)
    for iteration in range(args.iterations):
        idx = torch.randint(0, num_samples, (args.batch_size,))
        s = states[idx].to(device, non_blocking=True)
        ns = next_states[idx].to(device, non_blocking=True)
        c = cmd_states[idx].to(device, non_blocking=True) if use_command else None
        nc = (
            next_cmd_states[idx].to(device, non_blocking=True)
            if use_command
            else None
        )
        a = actions[idx].to(device, non_blocking=True)
        r = rewards[idx].to(device, non_blocking=True)
        d = dones[idx].to(device, non_blocking=True)

        with torch.no_grad():
            next_a, next_logp, _ = actor_sample(actor_target, ns, nc)
            target_q = r + args.gamma * (1.0 - d) * (
                critic_forward(critic_target, ns, nc, next_a) - args.alpha * next_logp
            )
        critic_prediction = critic_forward(critic, s, c, a)
        critic_loss = F.mse_loss(critic_prediction, target_q)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        last_critic_loss = critic_loss.detach().cpu()

        if iteration % args.policy_delay == 0:
            policy_action, logp, policy_mean = actor_sample(actor, s, c)
            q_value = critic_forward(critic, s, c, policy_action)
            actor_loss = (args.alpha * logp - q_value).mean() + args.bc_weight * F.mse_loss(policy_mean, a)
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            last_actor_loss = actor_loss.detach().cpu()
            soft_update(actor_target, actor, args.tau)

        soft_update(critic_target, critic, args.tau)

    best_index = int(torch.argmax(final_coverage).item())
    reference_images = state_image[best_index]
    reference_states = cmd_state[best_index]
    reference_command_raw = command_state[best_index].float()
    reference_commands = cmd_norm.encode(reference_command_raw)
    reference_goals = None if goal_cmd_pose is None else goal_cmd_pose[best_index].float()

    rollout_actions = []
    predicted_q = []
    action_min = actions_raw.min(0).values
    action_max = actions_raw.max(0).values

    actor.eval()
    critic.eval()
    with torch.no_grad():
        for step in range(trajectory_length):
            image = reference_images[step : step + 1].to(device)
            command = (
                reference_commands[step : step + 1].to(device)
                if use_command
                else None
            )
            action_encoded = actor_forward(actor, image, command, deterministic=True)
            decoded_action = action_norm.decode(action_encoded.cpu())
            decoded_action = torch.clamp(decoded_action, min=action_min, max=action_max)
            action_encoded = action_norm.encode(decoded_action).to(device)
            rollout_actions.append(decoded_action.squeeze(0))
            predicted_q.append(float(critic_forward(critic, image, command, action_encoded).cpu().item()))

    optimized = {
        "kind": "offline_rl_image_policy_optimized_trajectory",
        "source_file": args.input,
        "task": data.get("task", "Isaac-robot-US-reconstruction-v0"),
        "split": data.get("split"),
        "split_file": data.get("split_file"),
        "patient_ids": data.get("patient_ids"),
        "state_key": args.state_mode,
        "image_key": image_key,
        "policy": f"pi(a|s), s={args.state_mode}, tanh-Gaussian CNN actor, algorithm={args.algorithm}",
        "cmd_state": reference_states.unsqueeze(0),
        "action": torch.stack(rollout_actions, dim=0).unsqueeze(0),
        "predicted_q": torch.tensor(predicted_q, dtype=torch.float32).unsqueeze(0),
        "trajectory_length": int(trajectory_length),
        "source_num_traj": int(num_traj),
        "source_best_index": best_index,
        "source_best_final_coverage": float(final_coverage[best_index].item()),
        "source_mean_final_coverage": float(final_coverage.mean().item()),
        "training": {
            **vars(args),
            "state_key": args.state_mode,
            "state_mode": args.state_mode,
            "image_key": image_key,
            "image_shape": list(state_image.shape[2:]),
            "input_channels": int(input_channels),
            "cmd_state_dim": int(command_dim),
            "action_dim": action_dim,
            "num_transitions": int(num_samples),
            "reward": reward_description,
            "algorithm": args.algorithm,
            "last_actor_loss": float(last_actor_loss.item()),
            "last_critic_loss": float(last_critic_loss.item()),
        },
        "normalizers": {
            "action": {"mean": action_norm.mean, "std": action_norm.std},
            "cmd_state": {"mean": cmd_norm.mean, "std": cmd_norm.std},
        },
        "action_bounds": {
            "min": action_min,
            "max": action_max,
        },
        "policy_state": {
            "actor_state_dict": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
            "action_limit": action_limit.detach().cpu(),
            "hidden_dim": int(args.hidden_dim),
            "feature_dim": int(args.feature_dim),
            "action_dim": int(action_dim),
            "state_mode": args.state_mode,
            "image_key": image_key,
            "input_channels": int(input_channels),
            "cmd_state_dim": int(command_dim),
        },
    }
    optimized[image_key] = reference_images.unsqueeze(0)
    if "anatomy_prior" in data:
        optimized["anatomy_prior"] = data.get("anatomy_prior")
    if "anatomy_prior_volume" in data:
        optimized["anatomy_prior_volume"] = data.get("anatomy_prior_volume")
    if image_key != "us_image" and "us_image" in data:
        optimized["us_image"] = data["us_image"][best_index].unsqueeze(0)
    if reference_goals is not None:
        optimized["goal_cmd_pose"] = reference_goals.unsqueeze(0)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(optimized, args.output)
    print(f"[RESULT] saved optimized trajectory to {args.output}")
    print(f"[RESULT] policy=pi(a|s), state={args.state_mode}")
    print(f"[RESULT] source_num_traj={num_traj}")
    print(f"[RESULT] trajectory_length={trajectory_length}")
    print(f"[RESULT] image_key={image_key}")
    print(f"[RESULT] image_shape={tuple(state_image.shape[2:])}")
    print(f"[RESULT] source_best_index={best_index}")
    print(f"[RESULT] source_best_final_coverage={optimized['source_best_final_coverage']:.6f}")
    print(f"[RESULT] source_mean_final_coverage={optimized['source_mean_final_coverage']:.6f}")
    print(f"[RESULT] predicted_q_mean={optimized['predicted_q'].mean().item():.6f}")
    print(f"[RESULT] last_actor_loss={optimized['training']['last_actor_loss']:.6f}")
    print(f"[RESULT] last_critic_loss={optimized['training']['last_critic_loss']:.6f}")


if __name__ == "__main__":
    main()

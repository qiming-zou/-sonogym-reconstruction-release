"""Bounded smoke test for a single SonoGym task."""

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke test a SonoGym task.")
parser.add_argument("--task", required=True, help="Gymnasium task id to test.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps", type=int, default=2, help="Number of environment steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import sys
import torch

from isaaclab_tasks.utils import parse_env_cfg

import spinal_surgery  # noqa: F401  Registers SonoGym tasks.


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"SONOGYM_SMOKE_CREATED task={args_cli.task}", flush=True)
    try:
        try:
            reset_result = env.reset()
        except BaseException as exc:
            print(f"SONOGYM_SMOKE_RESET_EXCEPTION task={args_cli.task} exc={type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            raise
        print(f"SONOGYM_SMOKE_RESET task={args_cli.task}", flush=True)
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        action_dim = getattr(env.unwrapped, "num_actions", None)
        if action_dim is None:
            action_dim = env.action_space.shape[0]
        actions = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
        with torch.inference_mode():
            for step_id in range(args_cli.steps):
                obs, rew, terminated, truncated, info = env.step(actions)
                print(f"SONOGYM_SMOKE_STEP task={args_cli.task} step={step_id + 1}", flush=True)
        obs_desc = sorted(obs.keys()) if isinstance(obs, dict) else type(obs).__name__
        print(f"SONOGYM_SMOKE_OK task={args_cli.task} obs={obs_desc}", flush=True)
    finally:
        sys.stdout.flush()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

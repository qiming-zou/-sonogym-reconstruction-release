"""Install Isaac Lab dependencies and verify SonoGym task registration."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


NVIDIA_INDEX_URL = "https://pypi.nvidia.com"
PYTORCH_CU128_INDEX_URL = "https://download.pytorch.org/whl/cu128"
PYTORCH_CU130_INDEX_URL = "https://download.pytorch.org/whl/cu130"


def _run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def _isaaclab_spec() -> str:
    version = (sys.version_info.major, sys.version_info.minor)
    if version == (3, 10):
        return "isaaclab[isaacsim,all]==2.0.0"
    if version == (3, 11):
        return "isaaclab[isaacsim,all]==2.3.2.post1"
    raise RuntimeError(
        "Unsupported Python version for automatic Isaac Lab bootstrap: "
        f"{version[0]}.{version[1]}. Use Python 3.10 for Isaac Sim 4.5/Isaac Lab 2.0, "
        "or Python 3.11 for Isaac Sim 5.x/Isaac Lab 2.3."
    )


def _torch_install_args() -> tuple[list[str], str]:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return ["torch==2.7.0", "torchvision==0.22.0"], PYTORCH_CU128_INDEX_URL
    if machine in {"aarch64", "arm64"}:
        return ["torch==2.9.0", "torchvision==0.24.0"], PYTORCH_CU130_INDEX_URL
    raise RuntimeError(f"Unsupported architecture for CUDA PyTorch bootstrap: {machine}")


def install_dependencies(skip_torch: bool = False) -> None:
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            _isaaclab_spec(),
            "--extra-index-url",
            NVIDIA_INDEX_URL,
        ]
    )
    if not skip_torch:
        torch_specs, torch_index = _torch_install_args()
        _run([sys.executable, "-m", "pip", "install", "-U", *torch_specs, "--index-url", torch_index])


def verify_task_registration(accept_eula: bool = False) -> None:
    env = os.environ.copy()
    if accept_eula:
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"

    code = r"""
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import spinal_surgery  # noqa: F401

env_ids = sorted(spec.id for spec in gym.registry.values() if spec.id.startswith("Isaac-robot-US-reconstruction"))
print("SONOGYM_REGISTERED_TASKS=" + ",".join(env_ids))
if not env_ids:
    raise SystemExit("No SonoGym reconstruction tasks were registered.")

from sonogym_reconstruction_data import assets_data_dir
print("SONOGYM_ASSETS_DATA_DIR=" + assets_data_dir())

app.close()
"""
    _run([sys.executable, "-c", code], env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="Only verify the current environment.")
    parser.add_argument("--skip-torch", action="store_true", help="Do not install CUDA PyTorch wheels.")
    parser.add_argument(
        "--accept-nvidia-eula",
        action="store_true",
        help="Set OMNI_KIT_ACCEPT_EULA=YES for the verification run.",
    )
    parser.add_argument(
        "--write-env",
        type=Path,
        default=None,
        help="Write reusable environment exports for shells that run SonoGym.",
    )
    args = parser.parse_args(argv)

    if not args.skip_install:
        install_dependencies(skip_torch=args.skip_torch)

    verify_task_registration(accept_eula=args.accept_nvidia_eula)

    if args.write_env:
        args.write_env.write_text("export OMNI_KIT_ACCEPT_EULA=YES\n", encoding="utf-8")
        print(f"Wrote {args.write_env}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

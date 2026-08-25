"""Render a saved DiffStitch training sample as an offline video."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


parser = argparse.ArgumentParser(description="Visualize a DiffStitch sample saved as .pt.")
parser.add_argument("--input", type=str, default="artifacts/trajectories/diffstitch_single_sample.pt")
parser.add_argument("--output", type=str, default="artifacts/videos/diffstitch_single_sample.mp4")
parser.add_argument("--trajectory_index", type=int, default=0)
parser.add_argument("--fps", type=int, default=12)
parser.add_argument("--stride", type=int, default=2)
parser.add_argument("--dpi", type=int, default=120)
args = parser.parse_args()


SEGMENT_COLORS = {
    0: "#5aa9e6",
    1: "#ffbf3f",
    2: "#55c97a",
}
SEGMENT_NAMES = {
    0: "prefix",
    1: "diffusion bridge",
    2: "suffix",
}


def _to_rgb_us(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image = (image - image.min()) / max(float(image.max() - image.min()), 1e-6)
    rgb = np.stack([image, image, image], axis=-1)
    return rgb


def _draw_segmented_path(ax, xy: np.ndarray, segment_id: np.ndarray, end_step: int) -> None:
    upto = min(end_step + 1, xy.shape[0])
    for segment in (0, 1, 2):
        idx = np.where(segment_id[:upto] == segment)[0]
        if idx.size == 0:
            continue
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for run in runs:
            ax.plot(xy[run, 0], xy[run, 1], color=SEGMENT_COLORS[segment], linewidth=2.0)
    ax.scatter(xy[upto - 1, 0], xy[upto - 1, 1], color="#ff4d4d", s=24, zorder=5)


def _make_frame(data: dict, traj_idx: int, step: int, dpi: int) -> np.ndarray:
    us_image = data["us_image"][traj_idx, step].detach().cpu().numpy()
    cmd = data["cmd_state"][traj_idx].detach().cpu().numpy()
    action = data["action"][traj_idx].detach().cpu().numpy()
    segment_id = data.get("source_segment_id", torch.zeros(cmd.shape[0], dtype=torch.long).unsqueeze(0))[traj_idx]
    segment_id = segment_id.detach().cpu().numpy()
    reward_key = data.get("diffstitch", {}).get("reward_key", "prior_gain_reward")
    reward = data.get(reward_key, data.get("proxy_reward"))[traj_idx].detach().cpu().numpy()
    cumulative_reward = np.cumsum(np.clip(reward, 0.0, None))
    current_segment = int(segment_id[step])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_facecolor("#111418")
    for ax in axes.ravel():
        ax.set_facecolor("#111418")
        ax.tick_params(colors="#cad2dc", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#303946")

    axes[0, 0].imshow(_to_rgb_us(us_image), cmap="gray")
    axes[0, 0].set_title(f"US frame | {SEGMENT_NAMES.get(current_segment, 'segment')}", color="#eef3f8")
    axes[0, 0].axis("off")

    xy = cmd[:, [0, 1]]
    _draw_segmented_path(axes[0, 1], xy, segment_id, step)
    axes[0, 1].set_title("command path x-z", color="#eef3f8")
    axes[0, 1].set_xlabel("cmd x", color="#cad2dc")
    axes[0, 1].set_ylabel("cmd z", color="#cad2dc")
    axes[0, 1].grid(color="#303946", linewidth=0.6, alpha=0.8)
    axes[0, 1].set_aspect("equal", adjustable="datalim")

    time = np.arange(step + 1)
    for dim in range(action.shape[1]):
        axes[1, 0].plot(time, action[: step + 1, dim], linewidth=1.3, label=f"a{dim}")
    axes[1, 0].set_title("actions", color="#eef3f8")
    axes[1, 0].set_xlabel("step", color="#cad2dc")
    axes[1, 0].grid(color="#303946", linewidth=0.6, alpha=0.8)
    axes[1, 0].legend(loc="upper right", fontsize=7, frameon=False, labelcolor="#cad2dc")

    axes[1, 1].plot(cumulative_reward[: step + 1], color="#69db7c", linewidth=2.0)
    axes[1, 1].fill_between(time, cumulative_reward[: step + 1], color="#69db7c", alpha=0.16)
    axes[1, 1].set_title(f"cumulative {reward_key}", color="#eef3f8")
    axes[1, 1].set_xlabel("step", color="#cad2dc")
    axes[1, 1].grid(color="#303946", linewidth=0.6, alpha=0.8)

    for segment in (0, 1, 2):
        axes[1, 1].plot([], [], color=SEGMENT_COLORS[segment], label=SEGMENT_NAMES[segment])
    axes[1, 1].legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#cad2dc")

    fig.suptitle(
        f"DiffStitch sample | step {step:03d}/{cmd.shape[0] - 1:03d}",
        color="#f5f8fb",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def main() -> None:
    data = torch.load(args.input, map_location="cpu")
    if "us_image" not in data:
        raise KeyError("Input sample does not contain `us_image`.")
    num_traj = data["us_image"].shape[0]
    if args.trajectory_index >= num_traj:
        raise IndexError(f"trajectory_index={args.trajectory_index} but only {num_traj} trajectories exist.")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    trajectory_length = int(data["us_image"].shape[1])
    stride = max(1, int(args.stride))
    frame_steps = list(range(0, trajectory_length, stride))
    if frame_steps[-1] != trajectory_length - 1:
        frame_steps.append(trajectory_length - 1)

    with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for step in frame_steps:
            writer.append_data(_make_frame(data, args.trajectory_index, step, args.dpi))

    print(f"[RESULT] saved DiffStitch video to {args.output}")
    print(f"[RESULT] frames={len(frame_steps)} trajectory_length={trajectory_length}")


if __name__ == "__main__":
    main()

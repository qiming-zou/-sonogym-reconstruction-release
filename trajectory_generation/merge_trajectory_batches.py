"""Merge trajectory batch files along the trajectory dimension."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


parser = argparse.ArgumentParser(description="Merge saved trajectory batches.")
parser.add_argument("--inputs", nargs="+", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--split", type=str, default=None)
args = parser.parse_args()


CONCAT_KEYS = (
    "cmd_state",
    "goal_cmd_pose",
    "action",
    "reconstructed_volume_voxels",
    "delta_reconstructed_volume_voxels",
    "reconstructed_volume",
    "delta_reconstructed_volume",
    "prior_gain_reward",
    "proxy_reward",
    "final_coverage",
    "us_image",
    "us_ct_image",
)


def main() -> None:
    batches = [torch.load(path, map_location="cpu") for path in args.inputs]
    output = dict(batches[0])
    for key in CONCAT_KEYS:
        values = [batch.get(key) for batch in batches]
        if all(value is not None for value in values):
            output[key] = torch.cat(values, dim=0)
        elif any(value is not None for value in values):
            raise ValueError(f"Key `{key}` is present in only some batches.")
        else:
            output[key] = None

    output["num_traj"] = int(output["cmd_state"].shape[0])
    output["trajectory_length"] = int(output["cmd_state"].shape[1])
    output["mean_final_coverage"] = float(output["final_coverage"].float().mean().item())
    output["batch_files"] = [str(path) for path in args.inputs]
    output["batch_params"] = [batch.get("params", {}) for batch in batches]
    if args.split is not None:
        output["split"] = args.split
    patient_ids = []
    for batch in batches:
        for patient_id in batch.get("patient_ids", []):
            if patient_id not in patient_ids:
                patient_ids.append(patient_id)
    if patient_ids:
        output["patient_ids"] = patient_ids
    trajectory_patient_ids = []
    for batch in batches:
        trajectory_patient_ids.extend(batch.get("trajectory_patient_ids", []))
    if trajectory_patient_ids:
        output["trajectory_patient_ids"] = trajectory_patient_ids

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    print(f"[RESULT] saved {output['num_traj']} trajectories to {output_path}")
    print(f"[RESULT] trajectory_length={output['trajectory_length']}")
    print(f"[RESULT] mean_final_coverage={output['mean_final_coverage']:.6f}")
    for key in ("us_image", "prior_gain_reward", "final_coverage"):
        value = output.get(key)
        if value is not None:
            print(f"[RESULT] {key}_shape={tuple(value.shape)}")


if __name__ == "__main__":
    main()

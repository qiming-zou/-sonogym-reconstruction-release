"""Merge replay evaluation JSON files from patient batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description="Merge replay evaluation batches.")
parser.add_argument("--inputs", nargs="+", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--split", type=str, default=None)
args = parser.parse_args()


def main() -> None:
    batches = []
    for path in args.inputs:
        with Path(path).open("r") as f:
            batches.append(json.load(f))

    patient_ids = []
    final_per_env = []
    max_per_env = []
    mean_reward_per_env = []
    by_patient = {}
    for batch in batches:
        patient_ids.extend(batch["patient_ids"])
        final_per_env.extend(batch["replay_final_coverage_per_env"])
        max_per_env.extend(batch["replay_max_coverage_per_env"])
        mean_reward_per_env.extend(batch["replay_mean_reward_per_env"])
        by_patient.update(batch["replay_final_coverage_by_patient"])

    output = dict(batches[0])
    output.update(
        {
            "status": "replayed_in_isaac_batched",
            "split": args.split if args.split is not None else batches[0].get("split"),
            "num_envs": len(patient_ids),
            "batch_files": [str(path) for path in args.inputs],
            "patient_ids": patient_ids,
            "replay_final_coverage_per_env": final_per_env,
            "replay_final_coverage_by_patient": by_patient,
            "replay_final_coverage": sum(final_per_env) / max(1, len(final_per_env)),
            "replay_max_coverage_per_env": max_per_env,
            "replay_max_coverage": sum(max_per_env) / max(1, len(max_per_env)),
            "replay_mean_reward_per_env": mean_reward_per_env,
            "replay_mean_reward": sum(mean_reward_per_env) / max(1, len(mean_reward_per_env)),
            "coverage_trace": [],
            "coverage_trace_per_env": [],
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"[RESULT] saved merged replay eval to {output_path}")
    print(f"[RESULT] replay_final_coverage_mean={output['replay_final_coverage']:.6f}")
    print(f"[RESULT] patient_ids={patient_ids}")
    print(f"[RESULT] replay_final_coverage_by_patient={by_patient}")


if __name__ == "__main__":
    main()

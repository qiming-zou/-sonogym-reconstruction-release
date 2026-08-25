# Trajectory Generation

This folder contains the retained algorithm pipeline:

```text
random rollouts -> anatomy-prior reward -> offline RL policy -> online NBV replay
```

The anatomy prior is used for reward computation and online NBV goal selection.
The online NBV planner tracks its own prior-space visit mask instead of reading
the label-updated target reconstruction volume for planning. The prior is not
used as a policy observation channel.

Kept scripts:

- `build_anatomy_prior.py`: builds the target-organ prior from train patients.
- `generate_reconstruction_trajectories.py`: saves random rollouts with
  `us_image`, `cmd_state`, `goal_cmd_pose`, actions, and rewards.
- `extract_registration_training_priors.py`: runs training-patient rollouts and
  saves Open3D+probreg registration-prior snapshots from online sparse
  reconstructions.
- `merge_trajectory_batches.py`: merges train rollout batches.
- `train_offline_rl_reconstruction_policy.py`: trains the retained
  `us_image_goal_cmd` policy.
- `diffstitch_single_sample.py`: trains a small conditional diffusion bridge
  model and saves one stitched training sample for inspection.
- `replay_offline_rl_reconstruction.py`: evaluates the policy in Isaac with
  online NBV goals.
- `merge_replay_eval_batches.py`: merges multi-patient replay JSON files.

## Split

`data_splits/reconstruction_patients.json` defines:

```text
train: s0004, s0006, s0010, s0012, s0014, s0015, s0024, s0028
test:  s0029, s0030, s0034, s0038
```

Use the train split for prior construction, random rollout generation, and
offline RL training. Use the test split only for held-out replay.

## Retained State And Reward

The policy state is:

```text
s = [us_image, cur_cmd_state, goal_cmd_pose, goal_delta]
```

The saved ultrasound channel is the generated SonoGym `USSlicer.us_img_tensor`.

The retained reward is:

```text
prior_gain_reward = P_t - P_{t-1}
```

where `P_t` is the anatomy-prior probability mass covered by the current
reconstruction.

## Commands

Build the prior:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/build_anatomy_prior.py --split train --output artifacts/checkpoints/anatomy_prior_l4_train.pt
```

Generate random train batches. Use one patient per Isaac run; multi-patient
parallel generation can make later environments miss contact and distort the
saved rollouts.

```bash
for pid in s0004 s0006 s0010 s0012 s0014 s0015 s0024 s0028; do
  CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/generate_reconstruction_trajectories.py --task Isaac-robot-US-reconstruction-v0 --patient_ids "$pid" --num_envs 1 --num_traj 2 --trajectory_length 500 --goal_source random --min_final_coverage 0.0 --output "artifacts/trajectories/train_us_prior_batches/random_us_prior_train_${pid}.pt" --anatomy_prior artifacts/checkpoints/anatomy_prior_l4_train.pt --headless
done
```

Merge train batches:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/merge_trajectory_batches.py --inputs artifacts/trajectories/train_us_prior_batches/*.pt --output artifacts/trajectories/random_16_us_prior_train.pt --split train
```

Generate one expert trajectory for each original training sample. The retained
train set has two samples per train patient, so this creates 16 expert
trajectories with the same patient distribution.

```bash
for pid in s0004 s0006 s0010 s0012 s0014 s0015 s0024 s0028; do
  CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/generate_reconstruction_trajectories.py --task Isaac-robot-US-reconstruction-v0 --patient_ids "$pid" --num_envs 1 --num_traj 2 --trajectory_length 500 --goal_source expert --min_final_coverage 0.0 --output "artifacts/trajectories/expert_us_prior_batches/expert_us_prior_train_${pid}.pt" --anatomy_prior artifacts/checkpoints/anatomy_prior_l4_train.pt --headless
done
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/merge_trajectory_batches.py --inputs artifacts/trajectories/expert_us_prior_batches/*.pt --output artifacts/trajectories/expert_16_us_prior_train.pt --split train
```

Generate an AUS-SLAM active-reconstruction trajectory. This uses the anatomy
prior as the target belief, maintains online visited/uncertainty/frontier maps,
and chooses the next command goal by expected information gain.

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/generate_reconstruction_trajectories.py --task Isaac-robot-US-reconstruction-v0 --patient_ids s0004 --num_envs 1 --num_traj 1 --trajectory_length 500 --goal_source aus_slam --min_final_coverage 0.0 --output artifacts/trajectories/aus_slam_s0004.pt --anatomy_prior artifacts/checkpoints/anatomy_prior_l4_train.pt --headless
```

Generate the coverage-first receding-horizon AUS-SLAM trajectory. This version
plans a short sequence of informative scan goals and executes the horizon
endpoint, prioritizing maximum 500-step reconstruction coverage.

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/generate_reconstruction_trajectories.py --task Isaac-robot-US-reconstruction-v0 --patient_ids s0004 --num_envs 1 --num_traj 1 --trajectory_length 500 --goal_source aus_rh_slam --min_final_coverage 0.0 --output artifacts/trajectories/aus_rh_slam_s0004_500.pt --anatomy_prior artifacts/checkpoints/anatomy_prior_l4_train.pt --headless
```

Extract registration-prior snapshots for every training sample. This runs one
patient per Isaac process, uses online sparse reconstruction as the partial
point cloud, and saves both per-patient files and one merged file.

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/extract_registration_training_priors.py --split train --steps 500 --snapshot_steps 50,100,150,200,300,400,500 --goal_source random --anatomy_prior artifacts/checkpoints/anatomy_prior_l4_train.pt --output_dir artifacts/priors/registration_train --merged_output artifacts/priors/registration_train_all.pt --headless
```

Train the policy:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/train_offline_rl_reconstruction_policy.py --input artifacts/trajectories/random_16_us_prior_train.pt --output artifacts/trajectories/offline_rl_apo_us_goal_cmd_train.pt --iterations 2500 --batch_size 128 --algorithm sac_bc --state_mode us_image_goal_cmd --reward_key prior_gain_reward --device auto
```

Generate one DiffStitch-style training sample and render it without Isaac:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/diffstitch_single_sample.py --input artifacts/trajectories/random_16_us_prior_train.pt --output artifacts/trajectories/diffstitch_single_sample.pt --iterations 300 --bridge_length 32 --prefix_steps 64 --suffix_steps 96 --device auto
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p visualization/visualize_diffstitch_sample.py --input artifacts/trajectories/diffstitch_single_sample.pt --output artifacts/videos/diffstitch_single_sample.mp4 --stride 2
```

Replay held-out patients with online NBV goals, also one patient per Isaac run:

```bash
for pid in s0029 s0030 s0034 s0038; do
  CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/replay_offline_rl_reconstruction.py --task Isaac-robot-US-reconstruction-v0 --trajectory artifacts/trajectories/offline_rl_apo_us_goal_cmd_train.pt --patient_ids "$pid" --output_json "artifacts/trajectories/test_nbv_eval_batches/offline_rl_apo_us_goal_cmd_nbv_test_eval_${pid}.json" --mode policy --goal_source nbv --max_steps 500 --num_envs 1 --headless
done
```

Merge held-out replay batches:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/merge_replay_eval_batches.py --split test --inputs artifacts/trajectories/test_nbv_eval_batches/*.json --output artifacts/trajectories/offline_rl_apo_us_goal_cmd_nbv_eval_test.json
```

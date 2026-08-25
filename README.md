# SonoGym Reconstruction Only

This folder keeps one reconstruction algorithm:

```text
US image + goal_cmd_pose + online next-best-view planner + SAC_BC offline RL
```

The anatomy prior is still used by the online planner and by the
`prior_gain_reward`, but it is no longer passed to the policy as an image
observation channel.

The closed-loop policy is:

```text
pi(a | us_image, cur_cmd_state, goal_cmd_pose, goal_delta)
```

`goal_cmd_pose` is generated online by the NBV planner from the anatomy prior,
the planner's own prior-space visit mask, current probe pose, and reachable
command range. It does not use the label-updated target reconstruction volume
for planning.

## Kept Structure

```text
IsaacLab/                         local IsaacLab runtime
source/spinal_surgery/            reconstruction task, assets, sensors, controllers
visualization/                    Sono-style video visualization
trajectory_generation/            data generation, anatomy prior, offline RL, NBV replay
artifacts/checkpoints/            anatomy-prior checkpoint
artifacts/trajectories/           random train data, trained policy, NBV eval
docs/                             project notes
```

## Runtime

Use the local IsaacLab launcher:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p <script.py> [args...]
```

Install the extension in that Python environment:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p -m pip install -e source/spinal_surgery
```

For pip packaging, the stable runtime lives in `source/spinal_surgery` and is
published as `sonogym-reconstruction-core`; prior extraction and SLAM design
remain in `trajectory_generation/` or an external algorithm repo. The core
package exposes `spinal_surgery.interfaces.PriorExtractor` and
`spinal_surgery.interfaces.SLAMPlanner` so new algorithms can be implemented by
inheritance after `pip install`. The default pip release depends on
`sonogym-reconstruction-data`, which downloads the hosted data archive on first
use and exposes a single local `assets/data` directory automatically.
Developers can still override the asset path with:

```bash
export SONOGYM_ASSETS_DATA_DIR=/path/to/spinal_surgery/assets/data
```

See `docs/pip_core_split.md` for the package boundary and upload workflow, and
`docs/algorithm_interfaces.md` for prior/SLAM plugin templates. The
`sonogym-data` CLI remains an advanced fallback for private archive-based
deployments; the normal pip workflow does not require it. See
`docs/data_distribution.md`.

To bootstrap a fresh official pip IsaacLab environment and verify SonoGym task
registration after installing the package:

```bash
sonogym-bootstrap --accept-nvidia-eula
```

The bootstrap command follows NVIDIA's official pip installation path: it
installs IsaacLab/IsaacSim from `https://pypi.nvidia.com`, installs CUDA
PyTorch, accepts the EULA for the verification run, downloads SonoGym data on
first use, launches IsaacLab headless, and checks that
`Isaac-robot-US-reconstruction-v0` is registered.

## Data Split

The fixed split is stored in
`trajectory_generation/data_splits/reconstruction_patients.json`.

Train patients:

```text
s0004, s0006, s0010, s0012, s0014, s0015, s0024, s0028
```

Test patients:

```text
s0029, s0030, s0034, s0038
```

## Current Pipeline

Build the training anatomy prior:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/build_anatomy_prior.py \
  --split train \
  --output artifacts/checkpoints/anatomy_prior_l4_train.pt
```

Generate random training trajectories with real SonoGym
ultrasound observations, command state, goal command pose, and
`prior_gain_reward = P_t - P_{t-1}`. Random goals are sampled in the
approximate target-organ command range. Generate each patient with `num_envs=1`;
multi-patient parallel generation can make later environments miss contact and
distort the saved rollouts.

```bash
for pid in s0004 s0006 s0010 s0012 s0014 s0015 s0024 s0028; do
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/generate_reconstruction_trajectories.py \
  --task Isaac-robot-US-reconstruction-v0 \
  --patient_ids "$pid" \
  --num_envs 1 \
  --num_traj 2 \
  --trajectory_length 500 \
  --goal_source random \
  --min_final_coverage 0.0 \
  --output "artifacts/trajectories/train_us_prior_batches/random_us_prior_train_${pid}.pt" \
  --anatomy_prior artifacts/checkpoints/anatomy_prior_l4_train.pt \
  --headless
done
```

Merge generated batches:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/merge_trajectory_batches.py \
  --inputs artifacts/trajectories/train_us_prior_batches/*.pt \
  --output artifacts/trajectories/random_16_us_prior_train.pt \
  --split train
```

Train the retained offline RL policy:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/train_offline_rl_reconstruction_policy.py \
  --input artifacts/trajectories/random_16_us_prior_train.pt \
  --output artifacts/trajectories/offline_rl_apo_us_goal_cmd_train.pt \
  --iterations 2500 \
  --batch_size 128 \
  --algorithm sac_bc \
  --state_mode us_image_goal_cmd \
  --reward_key prior_gain_reward \
  --device auto
```

Replay on held-out test patients with online NBV goals, also one patient per
Isaac run:

```bash
for pid in s0029 s0030 s0034 s0038; do
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/replay_offline_rl_reconstruction.py \
  --task Isaac-robot-US-reconstruction-v0 \
  --trajectory artifacts/trajectories/offline_rl_apo_us_goal_cmd_train.pt \
  --patient_ids "$pid" \
  --output_json "artifacts/trajectories/test_nbv_eval_batches/offline_rl_apo_us_goal_cmd_nbv_test_eval_${pid}.json" \
  --mode policy \
  --goal_source nbv \
  --max_steps 500 \
  --num_envs 1 \
  --headless
done
```

Merge test batches:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/merge_replay_eval_batches.py \
  --split test \
  --inputs artifacts/trajectories/test_nbv_eval_batches/*.json \
  --output artifacts/trajectories/offline_rl_apo_us_goal_cmd_nbv_eval_test.json
```

Current retained 500-step results after replacing the training data with random
trajectories, retraining, and replaying in Isaac with online NBV goals:

```text
trajectory_length = 500
max_steps = 500
train trajectory mean final coverage = 0.518922
train replay mean final coverage     = 0.344647
test replay mean final coverage      = 0.280269
```

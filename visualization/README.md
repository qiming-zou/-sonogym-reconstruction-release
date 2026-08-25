# Visualization

This folder contains video visualization code.

- Main script: `run_expert_reconstruction_video.py`
- Default video folder: `artifacts/videos/`
- Default thumbnail folder: `artifacts/thumbnails/`

Run from the project root:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p visualization/run_expert_reconstruction_video.py --task Isaac-robot-US-reconstruction-v0 --num_envs 1 --steps 500 --capture_interval 5 --fps 15 --layout site --render_scene --headless
```

Use the same video layout for one online NBV trajectory:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p visualization/run_expert_reconstruction_video.py --task Isaac-robot-US-reconstruction-v0 --num_envs 1 --steps 500 --capture_interval 5 --fps 15 --layout site --render_scene --trajectory_source nbv --anatomy_prior artifacts/checkpoints/anatomy_prior_l4_train.pt --headless
```

Visualize one policy replay trajectory on a held-out test patient:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p visualization/run_expert_reconstruction_video.py --task Isaac-robot-US-reconstruction-v0 --num_envs 1 --patient_ids s0029 --trajectory_source policy --trajectory artifacts/trajectories/offline_rl_apo_us_goal_cmd_train.pt --steps 500 --capture_interval 5 --fps 15 --layout site --render_scene --output_dir artifacts/videos/test_policy --thumbnail_dir artifacts/thumbnails/test_policy --headless
```

Execute and visualize one saved DiffStitch trajectory in SonoGym:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p visualization/run_expert_reconstruction_video.py --task Isaac-robot-US-reconstruction-v0 --num_envs 1 --patient_ids s0024 --trajectory_source stitching --trajectory artifacts/trajectories/diffstitch_single_sample.pt --trajectory_index 0 --steps 192 --capture_interval 5 --fps 12 --output_dir artifacts/videos/stitching --thumbnail_dir artifacts/thumbnails/stitching --headless
```

Execute and visualize the original random trajectory used as the DiffStitch prefix:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p visualization/run_expert_reconstruction_video.py --task Isaac-robot-US-reconstruction-v0 --num_envs 1 --patient_ids s0024 --trajectory_source random --trajectory artifacts/trajectories/random_16_us_prior_train.pt --trajectory_index 13 --steps 192 --capture_interval 5 --fps 12 --output_dir artifacts/videos/random --thumbnail_dir artifacts/thumbnails/random --headless
```

Replay one saved expert trajectory from the expert training set:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p visualization/run_expert_reconstruction_video.py --task Isaac-robot-US-reconstruction-v0 --num_envs 1 --patient_ids s0004 --trajectory_source expert_replay --trajectory artifacts/trajectories/expert_16_us_prior_train.pt --trajectory_index 0 --steps 500 --capture_interval 5 --fps 12 --output_dir artifacts/videos/expert_train_samples --thumbnail_dir artifacts/thumbnails/expert_train_samples --headless
```

Visualize online AUS-SLAM active reconstruction:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p visualization/run_expert_reconstruction_video.py --task Isaac-robot-US-reconstruction-v0 --num_envs 1 --patient_ids s0004 --trajectory_source aus_slam --steps 500 --capture_interval 5 --fps 12 --output_dir artifacts/videos/aus_slam --thumbnail_dir artifacts/thumbnails/aus_slam --anatomy_prior artifacts/checkpoints/anatomy_prior_l4_train.pt --headless
```

Visualize coverage-first receding-horizon AUS-SLAM:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p visualization/run_expert_reconstruction_video.py --task Isaac-robot-US-reconstruction-v0 --num_envs 1 --patient_ids s0004 --trajectory_source aus_rh_slam --steps 500 --capture_interval 5 --fps 12 --output_dir artifacts/videos/aus_rh_slam --thumbnail_dir artifacts/thumbnails/aus_rh_slam --anatomy_prior artifacts/checkpoints/anatomy_prior_l4_train.pt --headless
```

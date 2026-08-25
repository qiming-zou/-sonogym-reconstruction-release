# Reconstruction-Only Project Structure

The project is organized around one retained algorithm:

```text
goal-conditioned offline RL + online next-best-view goal planner
```

## visualization/

Sono-style video generation code for inspecting reconstruction rollouts.

- `run_expert_reconstruction_video.py`: renders robot, human body, ultrasound
  plane/image, reconstructed voxels, and 2D/3D trajectories.

## trajectory_generation/

The active training and evaluation pipeline.

- `build_anatomy_prior.py`: builds the train-patient anatomy prior.
- `generate_nbv_suboptimal_reconstruction_trajectories.py`: creates NBV-guided
  suboptimal ultrasound rollouts around the target organ range and rejects
  trajectories below the configured final coverage threshold.
- `merge_trajectory_batches.py`: merges generated rollout batches.
- `train_offline_rl_reconstruction_policy.py`: trains the retained
  `us_image_goal_cmd` offline RL policy.
- `replay_offline_rl_reconstruction.py`: evaluates that policy in Isaac using
  online NBV goal generation.
- `merge_replay_eval_batches.py`: merges held-out patient evaluation JSON files.
- `data_splits/reconstruction_patients.json`: fixed train/test split.

## artifacts/

Generated files. Code should not be stored here.

- `checkpoints/anatomy_prior_l4_train.pt`: retained train-patient anatomy prior.
- `trajectories/nbv_suboptimal_16_us_prior_train.pt`: retained training
  rollouts.
- `trajectories/offline_rl_apo_us_goal_cmd_train.pt`: retained trained
  policy.
- `trajectories/offline_rl_apo_us_goal_cmd_nbv_eval_test.json`: retained
  held-out NBV replay result.
- `videos/` and `thumbnails/`: visualization outputs when generated.

## docs/

Project-level notes.

- `PROJECT_STRUCTURE.md`: this file.
- `DO_NOT_TOUCH.md`: files and folders that should not be moved casually.

## Do Not Touch By Default

These are runtime/task infrastructure rather than experiment code:

- `IsaacLab/`
- `source/spinal_surgery/`
- `docker/`
- `tools/`
- package metadata and root dotfiles

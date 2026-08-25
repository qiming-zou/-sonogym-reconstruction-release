# Do Not Touch

These project areas are required by IsaacLab, the reconstruction task, assets, or package installation. Do not move or rewrite them while working on visualization or trajectory scripts.

- `IsaacLab/`: local IsaacLab runtime and launcher.
- `source/spinal_surgery/`: reconstruction task package, controllers, sensors, kinematics, configs, and assets.
- `docker/`: container/runtime support.
- `tools/`: mesh and utility conversion tools copied from SonoGym.
- `pyproject.toml`, `source/spinal_surgery/setup.py`, `source/spinal_surgery/pyproject.toml`: package/install metadata.
- root dotfiles such as `.gitignore`, `.flake8`, `.pre-commit-config.yaml`: tooling metadata.

Only edit these areas when the task itself, assets, or environment registration must change.

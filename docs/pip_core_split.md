# Core Package Split

This repository is split into two working layers.

## 1. Pip Core Layer

Package path:

```text
source/spinal_surgery
```

Pip distribution name:

```text
sonogym-reconstruction-core
```

Python import name remains:

```python
import spinal_surgery
```

This package contains the stable SonoGym reconstruction runtime:

- IsaacLab task registration.
- Robotic ultrasound reconstruction environment.
- Surface motion planner and local controller.
- Ultrasound/label slicers and surface reconstructor.
- Lightweight configs and robot/bed/probe assets.
- Abstract plugin interfaces:
  - `spinal_surgery.interfaces.PriorExtractor`
  - `spinal_surgery.interfaces.SLAMPlanner`

It intentionally does not contain the research-side prior/SLAM code from
`trajectory_generation/`.

## 2. Research Layer

Keep these modules outside the core pip package while iterating:

```text
trajectory_generation/active_us_slam_planner.py
trajectory_generation/registration_prior.py
trajectory_generation/patient_conditioned_prior.py
trajectory_generation/view_gain_prior.py
trajectory_generation/build_anatomy_prior.py
trajectory_generation/generate_reconstruction_trajectories.py
visualization/
```

This layer can depend on extra packages such as `open3d`, `probreg`, and custom
training code without forcing those dependencies into the core runtime package.

## Patient Data

For the default zero-extra-operation install, publish the companion data package
beside the core package:

```text
sonogym-reconstruction-data==0.1.1
```

`sonogym-reconstruction-core` depends on that package. The data package is a
small locator/downloader package:

```text
sonogym-reconstruction-data==0.1.1
```

The data package downloads a hosted HTTPS archive on first use and extracts it
as one local `assets/data` directory. Users only run:

```bash
pip install sonogym-reconstruction-core==0.1.1
```

For local development or custom datasets, override with:

```bash
export SONOGYM_ASSETS_DATA_DIR=/path/to/assets/data
```

See `docs/data_distribution.md` for archive hosting, checksum, and upload
instructions.

The directory should contain:

```text
HumanModels/selected_dataset/
HumanModels/selected_dataset_stl/
HumanModels/selected_dataset_body_from_urdf/
MedicalBed/
Robots/
SurgicalTools/
```

## Local Install

Inside the SonoGym/IsaacLab environment:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p -m pip install -e source/spinal_surgery
```

After publishing:

```bash
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p -m pip install sonogym-reconstruction-core
```

For a fresh official pip IsaacLab setup, run the installed bootstrap command in
the target Python environment:

```bash
sonogym-bootstrap --accept-nvidia-eula
```

The command installs IsaacLab/IsaacSim from NVIDIA's pip index, installs CUDA
PyTorch, triggers SonoGym data download, launches IsaacLab headless, and checks
that `Isaac-robot-US-reconstruction-v0` is registered. Use
`sonogym-bootstrap --skip-install --accept-nvidia-eula` when IsaacLab is already
installed and only registration should be verified.

Then develop prior/SLAM code in this repository against the installed core:

```bash
export SONOGYM_ASSETS_DATA_DIR=/home/lab/Desktop/sonogym_reconstruction_only/source/spinal_surgery/spinal_surgery/assets/data
CONDA_PREFIX=/home/lab/miniconda3/envs/sonogym ./IsaacLab/isaaclab.sh -p trajectory_generation/generate_reconstruction_trajectories.py --headless ...
```

For external algorithm development, inherit the core interfaces instead of
editing the pip package:

```python
from spinal_surgery.interfaces import PriorExtractor, SLAMPlanner
```

See `docs/algorithm_interfaces.md` for complete templates.

Do not use a bare `python -c "import spinal_surgery"` as the health check.
The package registers IsaacLab tasks at import time, so it must be imported
after Isaac Sim's `AppLauncher` has initialized the `omni` modules:

```python
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
import spinal_surgery
```

## Build And Upload

Build the hosted data archive:

```bash
python - <<'PY'
from pathlib import Path
import tarfile
root = Path("source/spinal_surgery/spinal_surgery/assets/data").resolve()
out = Path("dist/sonogym_reconstruction_data_assets_0.1.1.tar.gz").resolve()
out.parent.mkdir(exist_ok=True)
with tarfile.open(out, "w:gz") as tar:
    tar.add(root, arcname="assets/data")
print(out)
PY
```

Build release wheels:

```bash
python -m pip wheel source/sonogym_reconstruction_data --no-deps -w dist
python -m pip wheel source/spinal_surgery --no-deps -w dist
```

Upload to TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

Upload to PyPI:

```bash
python -m twine upload dist/*
```

Before uploading, inspect the wheel and ensure `HumanModels` is not included:

```bash
python -m zipfile -l dist/*.whl | grep HumanModels
```

The command should print nothing.

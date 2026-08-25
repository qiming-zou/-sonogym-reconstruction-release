# Data Distribution

The default release is designed so a user can run:

```bash
pip install sonogym-reconstruction-core
```

and get the full runtime plus data dependency chain:

```text
sonogym-reconstruction-core
sonogym-reconstruction-data
```

The core package automatically discovers the installed data package through
`sonogym_reconstruction_data.assets_data_dir()`. No environment variable or
manual copy is required for the default workflow.

Official PyPI currently has a default 100 MB limit for each uploaded file and a
10 GB total project limit
([PyPI storage limits](https://docs.pypi.org/project-management/storage-limits/)).
The SonoGym data tree is larger than one wheel can carry under that default, so
the public release keeps PyPI wheels small and hosts the full data archive on a
regular HTTPS file server. The data package downloads and verifies that archive
on first use, then exposes a local `assets/data` directory from the cache.

## Required Directory Layout

The assets root should look like:

```text
assets/data/
  HumanModels/
    selected_dataset/
    selected_dataset_stl/
    selected_dataset_body_from_urdf/
  MedicalBed/
  Robots/
  SurgicalTools/
```

The downloaded archive contains this complete `assets/data` tree. The core
package also contains lightweight robot/bed/probe assets for development, but
the full data package is the default runtime source.

## Build PyPI Wheels

Build the data archive:

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

Upload this archive to an HTTPS file server. Then set
`DEFAULT_DATA_URL` and `DEFAULT_DATA_SHA256` in
`source/sonogym_reconstruction_data/sonogym_reconstruction_data/__init__.py`.

Build the small data wheel and the core wheel:

```bash
python -m pip wheel source/sonogym_reconstruction_data --no-deps -w dist
python -m pip wheel source/spinal_surgery --no-deps -w dist
```

Upload only the wheels to PyPI. The large archive stays on the file server.
For internal deployments, the archive can be hosted on:

```text
GitHub Release
Hugging Face dataset
S3 / OSS / MinIO public object URL
any HTTPS static file host
```

If the wheels are available to pip, this is enough:

```bash
pip install sonogym-reconstruction-core
sonogym-bootstrap --accept-nvidia-eula
```

## Verify Install

After install:

```python
from sonogym_reconstruction_data import assets_data_dir
print(assets_data_dir())
```

The first call may create symlinks or hardlinks under
`~/.cache/sonogym_reconstruction_data/0.1.1/assets/data`. With URL-based data
distribution, it downloads the archive once, verifies sha256, and extracts it
there.

## Advanced: Override Data Path

Developers can override packaged data with:

```bash
export SONOGYM_ASSETS_DATA_DIR=/path/to/assets/data
```

To use a different hosted archive without rebuilding the wheel:

```bash
export SONOGYM_DATA_URL=https://your-host/sonogym_reconstruction_data_assets_0.1.1.tar.gz
export SONOGYM_DATA_SHA256=<sha256>
```

## Advanced: Manual Archive Download

The `sonogym-data` CLI remains available for slim/private deployments where the
data package is not installed:

```bash
sonogym-data --url https://your-host/sonogym_humanmodels_v0.1.0.tar.gz --target ~/sonogym_assets/data
```

Validate an existing directory:

```bash
sonogym-data --check --target ~/sonogym_assets/data
```

Use an already downloaded archive:

```bash
sonogym-data \
  --archive ~/Downloads/sonogym_humanmodels_v0.1.0.tar.gz \
  --sha256 <sha256> \
  --target ~/sonogym_assets/data
```

## Environment Variables

The CLI also accepts:

```bash
export SONOGYM_DATA_URL=https://your-host/sonogym_humanmodels_v0.1.0.tar.gz
export SONOGYM_DATA_SHA256=<sha256>
export SONOGYM_ASSETS_DATA_DIR=~/sonogym_assets/data
sonogym-data
```

## Current Status

The repository now contains:

- `source/sonogym_reconstruction_data`: the data locator package that downloads
  and verifies the hosted data archive.
- `tools/build_data_shards.py`: a fallback release script for shard-based
  package sources, kept for private wheelhouse deployments.

To guarantee zero extra user operations, publish
`sonogym-reconstruction-data==0.1.1` and `sonogym-reconstruction-core==0.1.1`
to the same pip index, and keep the configured HTTPS data archive online.

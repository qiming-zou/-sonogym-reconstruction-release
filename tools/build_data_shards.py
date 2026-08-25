"""Generate PyPI-sized SonoGym reconstruction data shard package sources."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


VERSION = "0.1.0"
DEFAULT_SHARD_COUNT = 32
DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "spinal_surgery"
    / "spinal_surgery"
    / "assets"
    / "data"
)


@dataclass
class Shard:
    index: int
    size: int = 0
    files: list[Path] = field(default_factory=list)


def _collect_files(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.rglob("*") if path.is_file())


def _assign_files(data_root: Path, files: list[Path], shard_count: int) -> list[Shard]:
    shards = [Shard(index=index) for index in range(shard_count)]
    for path in sorted(files, key=lambda item: item.stat().st_size, reverse=True):
        shard = min(shards, key=lambda item: item.size)
        shard.files.append(path.relative_to(data_root))
        shard.size += path.stat().st_size
    return shards


def _package_name(index: int) -> str:
    return f"sonogym_reconstruction_data_shard_{index:03d}"


def _distribution_name(index: int) -> str:
    return f"sonogym-reconstruction-data-shard-{index:03d}"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _link_data_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    try:
        os.link(source, destination)
        return
    except OSError:
        pass

    try:
        os.symlink(source, destination)
        return
    except OSError:
        pass

    shutil.copy2(source, destination)


def _setup_py(package_name: str, distribution_name: str) -> str:
    return f'''"""Installation script for {distribution_name}."""

from __future__ import annotations

import os
from setuptools import find_packages, setup


def _package_data():
    root = os.path.join(os.path.dirname(__file__), "{package_name}", "assets", "data")
    files = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            files.append(os.path.relpath(full_path, os.path.join(os.path.dirname(__file__), "{package_name}")))
    return files


setup(
    name="{distribution_name}",
    version="{VERSION}",
    description="SonoGym reconstruction data shard.",
    packages=find_packages(),
    package_data={{"{package_name}": _package_data()}},
    python_requires=">=3.10",
    zip_safe=False,
)
'''


def _pyproject_toml() -> str:
    return """[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"
"""


def _write_shard(data_root: Path, output_root: Path, shard: Shard) -> None:
    package_name = _package_name(shard.index)
    distribution_name = _distribution_name(shard.index)
    shard_root = output_root / package_name
    package_root = shard_root / package_name

    if shard_root.exists():
        shutil.rmtree(shard_root)

    _write_text(shard_root / "pyproject.toml", _pyproject_toml())
    _write_text(shard_root / "setup.py", _setup_py(package_name, distribution_name))
    _write_text(package_root / "__init__.py", f'"""SonoGym data shard {shard.index:03d}."""\n')

    for relative in shard.files:
        _link_data_file(data_root / relative, package_root / "assets" / "data" / relative)


def _format_mb(size: int) -> str:
    return f"{size / 1024 / 1024:.1f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("build/sonogym_data_shards"))
    parser.add_argument("--num-shards", type=int, default=DEFAULT_SHARD_COUNT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")

    files = _collect_files(data_root)
    shards = _assign_files(data_root, files, args.num_shards)
    largest = max(shards, key=lambda shard: shard.size)
    total_size = sum(shard.size for shard in shards)

    print(f"data_root={data_root}")
    print(f"files={len(files)} total={_format_mb(total_size)} shards={len(shards)}")
    print(f"largest_shard={_package_name(largest.index)} raw_size={_format_mb(largest.size)}")

    for shard in shards:
        print(f"{_package_name(shard.index)} files={len(shard.files)} raw_size={_format_mb(shard.size)}")

    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        _write_shard(data_root, output_root, shard)

    print(f"wrote shard package sources to {output_root}")
    print("build wheels with:")
    print(f"  for d in {output_root}/sonogym_reconstruction_data_shard_*; do python -m pip wheel \"$d\" --no-deps -w dist; done")


if __name__ == "__main__":
    main()

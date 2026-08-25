"""Locate and download SonoGym reconstruction patient assets.

The PyPI package is intentionally small.  Full patient assets are hosted as a
regular HTTPS archive and downloaded into a local cache on first use.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from importlib import resources
from pathlib import Path

__version__ = "0.1.1"

SHARD_COUNT = 32
SHARD_PACKAGE_PREFIX = "sonogym_reconstruction_data_shard_"
_CACHE_MARKER = ".sonogym_reconstruction_data_complete"
DEFAULT_DATA_URL = (
    "https://github.com/qiming-zou/-sonogym-reconstruction-release/releases/download/"
    "sonogym-reconstruction-v0.1.2/sonogym_reconstruction_data_assets_0.1.1.tar.gz"
)
DEFAULT_DATA_SHA256 = "c82f6a3445ae240eb20fb870a5cce2bf6915a45b2ea80c2035faa89c36a018d9"
_REQUIRED_RELATIVE_DIRS = (
    "HumanModels/selected_dataset",
    "HumanModels/selected_dataset_stl",
    "HumanModels/selected_dataset_body_from_urdf",
    "MedicalBed",
    "Robots",
    "SurgicalTools",
)


def _has_required_layout(path: Path) -> bool:
    return all((path / relative).is_dir() for relative in _REQUIRED_RELATIVE_DIRS)


def _cache_assets_dir() -> Path:
    cache_base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_base / "sonogym_reconstruction_data" / __version__ / "assets" / "data"


def _cache_root() -> Path:
    return _cache_assets_dir().parents[1]


def _local_assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets" / "data"


def _shard_assets_dirs() -> list[Path]:
    shard_dirs: list[Path] = []
    missing: list[str] = []
    for index in range(SHARD_COUNT):
        package_name = f"{SHARD_PACKAGE_PREFIX}{index:03d}"
        try:
            importlib.import_module(package_name)
        except ModuleNotFoundError as exc:
            if exc.name == package_name:
                missing.append(package_name)
                continue
            raise

        shard_dir = resources.files(package_name) / "assets" / "data"
        shard_path = Path(str(shard_dir))
        if not shard_path.is_dir():
            missing.append(f"{package_name}:assets/data")
            continue
        shard_dirs.append(shard_path)

    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} missing)"
        raise RuntimeError(
            "SonoGym reconstruction data shards are incomplete. "
            f"Missing {preview}{suffix}. Reinstall sonogym-reconstruction-core."
        )
    return shard_dirs


def _link_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        try:
            if destination.stat().st_size == source.stat().st_size:
                return
        except OSError:
            pass
        destination.unlink()

    try:
        os.symlink(source, destination)
        return
    except OSError:
        pass

    try:
        os.link(source, destination)
        return
    except OSError:
        pass

    shutil.copy2(source, destination)


def _materialize_shards() -> Path:
    target = _cache_assets_dir()
    marker = target.parent / _CACHE_MARKER
    if marker.exists() and _has_required_layout(target):
        return target

    shard_dirs = _shard_assets_dirs()
    target.mkdir(parents=True, exist_ok=True)
    for shard_dir in shard_dirs:
        for source in shard_dir.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(shard_dir)
            _link_or_copy_file(source, target / relative)

    if not _has_required_layout(target):
        raise RuntimeError(
            "SonoGym reconstruction data shards were installed, but the merged "
            f"assets directory is incomplete: {target}"
        )

    marker.write_text(f"version={__version__}\nshards={SHARD_COUNT}\n", encoding="utf-8")
    return target


def _download_url() -> str:
    return os.environ.get("SONOGYM_DATA_URL", DEFAULT_DATA_URL).strip()


def _expected_sha256() -> str:
    return os.environ.get("SONOGYM_DATA_SHA256", DEFAULT_DATA_SHA256).strip().lower()


def _archive_path() -> Path:
    return _cache_root() / "downloads" / f"sonogym_reconstruction_data_assets_{__version__}.tar.gz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()

    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as file:
        shutil.copyfileobj(response, file, length=1024 * 1024)

    expected = _expected_sha256()
    if expected:
        actual = _sha256(partial)
        if actual != expected:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                "Downloaded SonoGym data archive failed sha256 verification: "
                f"expected {expected}, got {actual}"
            )
    partial.replace(destination)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(destination_root) + os.sep):
                raise RuntimeError(f"Unsafe path in SonoGym data archive: {member.name}")
        tar.extractall(destination)


def _download_and_extract() -> Path:
    target = _cache_assets_dir()
    marker = target.parent / _CACHE_MARKER
    if marker.exists() and _has_required_layout(target):
        return target

    url = _download_url()
    if not url:
        raise RuntimeError(
            "SonoGym reconstruction data is not installed and no data URL is configured. "
            "Set SONOGYM_DATA_URL or install data shards."
        )

    archive = _archive_path()
    expected = _expected_sha256()
    if not archive.exists() or (expected and _sha256(archive) != expected):
        _download_archive(url, archive)

    with tempfile.TemporaryDirectory(prefix="sonogym_data_extract_") as tmp:
        tmp_root = Path(tmp)
        _safe_extract_tar(archive, tmp_root)
        extracted = tmp_root / "assets" / "data"
        if not _has_required_layout(extracted):
            raise RuntimeError(f"SonoGym data archive has an invalid layout: {archive}")

        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(target))

    marker.write_text(f"version={__version__}\nurl={url}\n", encoding="utf-8")
    return target


def assets_data_dir() -> str:
    """Return a complete ``assets/data`` directory.

    Resolution order:

    1. ``SONOGYM_ASSETS_DATA_DIR`` for custom or private datasets.
    2. A local package ``assets/data`` tree, used by editable/source installs.
    3. A materialized cache assembled automatically from installed shard wheels.
    4. A cache downloaded automatically from ``SONOGYM_DATA_URL`` or the built-in
       default data archive URL.
    """

    override = os.environ.get("SONOGYM_ASSETS_DATA_DIR")
    if override:
        return str(Path(override).expanduser().resolve())

    local_dir = _local_assets_dir()
    if _has_required_layout(local_dir):
        return str(local_dir)

    try:
        return str(_materialize_shards())
    except RuntimeError:
        return str(_download_and_extract())

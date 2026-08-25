"""Download and validate external SonoGym patient assets."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REQUIRED_RELATIVE_PATHS = [
    "HumanModels/selected_dataset",
    "HumanModels/selected_dataset_stl",
    "HumanModels/selected_dataset_body_from_urdf",
]


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, output.open("wb") as f:
        shutil.copyfileobj(response, f)


def extract_archive(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive.suffixes)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        return
    if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz") or archive.suffix == ".tar":
        with tarfile.open(archive) as tf:
            tf.extractall(target)
        return
    raise ValueError(f"Unsupported archive format: {archive}")


def find_assets_root(target: Path) -> Path:
    candidates = [target, target / "assets" / "data", target / "data"]
    for child in target.iterdir() if target.exists() else []:
        if child.is_dir():
            candidates.extend([child, child / "assets" / "data", child / "data"])
    for candidate in candidates:
        if all((candidate / rel).exists() for rel in REQUIRED_RELATIVE_PATHS):
            return candidate
    return target


def check_assets(target: Path) -> tuple[bool, list[str], Path]:
    root = find_assets_root(target)
    missing = [rel for rel in REQUIRED_RELATIVE_PATHS if not (root / rel).exists()]
    return not missing, missing, root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download or validate SonoGym external patient assets.")
    parser.add_argument("--url", type=str, default=os.environ.get("SONOGYM_DATA_URL"))
    parser.add_argument(
        "--target",
        type=str,
        default=os.environ.get("SONOGYM_ASSETS_DATA_DIR", "sonogym_assets/data"),
        help="Target assets/data directory.",
    )
    parser.add_argument("--sha256", type=str, default=os.environ.get("SONOGYM_DATA_SHA256"))
    parser.add_argument("--archive", type=str, default=None, help="Use an already downloaded .zip/.tar/.tar.gz archive.")
    parser.add_argument("--check", action="store_true", help="Only validate the target directory.")
    args = parser.parse_args(argv)

    target = Path(args.target).expanduser().resolve()
    if args.check:
        ok, missing, root = check_assets(target)
        if ok:
            print(f"[OK] SonoGym assets found at: {root}")
            print(f"export SONOGYM_ASSETS_DATA_DIR={root}")
            return 0
        print(f"[ERROR] SonoGym assets are incomplete under: {target}", file=sys.stderr)
        for rel in missing:
            print(f"missing: {rel}", file=sys.stderr)
        return 2

    if args.archive is None and not args.url:
        print(
            "No data URL was provided. Set SONOGYM_DATA_URL or pass --url after uploading "
            "the HumanModels archive to HuggingFace, Zenodo, S3, or an internal file server.",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="sonogym_data_") as tmp:
        if args.archive is not None:
            archive = Path(args.archive).expanduser().resolve()
        else:
            filename = Path(str(args.url).split("?")[0]).name or "sonogym_assets.tar.gz"
            archive = Path(tmp) / filename
            print(f"[INFO] downloading {args.url}")
            download(args.url, archive)

        if args.sha256:
            actual = sha256sum(archive)
            if actual.lower() != args.sha256.lower():
                print(f"[ERROR] sha256 mismatch: expected {args.sha256}, got {actual}", file=sys.stderr)
                return 3
            print("[OK] sha256 verified")

        print(f"[INFO] extracting {archive} -> {target}")
        extract_archive(archive, target)

    ok, missing, root = check_assets(target)
    if not ok:
        print(f"[ERROR] extracted assets are incomplete under: {target}", file=sys.stderr)
        for rel in missing:
            print(f"missing: {rel}", file=sys.stderr)
        return 4
    print(f"[OK] SonoGym assets installed at: {root}")
    print(f"export SONOGYM_ASSETS_DATA_DIR={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

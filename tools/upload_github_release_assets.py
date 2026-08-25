"""Upload SonoGym release assets to a GitHub Release without the gh CLI.

Required:
  GITHUB_TOKEN with repository contents/write permission.

Example:
  GITHUB_TOKEN=github_pat_xxx python tools/upload_github_release_assets.py \
    --repo owner/repo \
    --tag sonogym-reconstruction-v0.1.1 \
    --title "SonoGym Reconstruction v0.1.1"
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_ASSETS = [
    "dist/sonogym_reconstruction_core-0.1.3-py3-none-any.whl",
    "dist/sonogym_reconstruction_data-0.1.2-py3-none-any.whl",
    "dist/sonogym_reconstruction_data_assets_0.1.1.tar.gz",
]

DATA_INIT = Path("source/sonogym_reconstruction_data/sonogym_reconstruction_data/__init__.py")
DATA_ARCHIVE = Path("dist/sonogym_reconstruction_data_assets_0.1.1.tar.gz")


def _github_release_asset_url(repo: str, tag: str, asset_name: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"


def _rewrite_default_data_url(url: str) -> None:
    text = DATA_INIT.read_text(encoding="utf-8")
    replacement = f'DEFAULT_DATA_URL = "{url}"'
    text, count = re.subn(
        r'DEFAULT_DATA_URL\s*=\s*\([^)]*\)|DEFAULT_DATA_URL\s*=\s*"[^"]*"',
        replacement,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"Could not rewrite DEFAULT_DATA_URL in {DATA_INIT}")
    DATA_INIT.write_text(text, encoding="utf-8")


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _prepare_wheels(repo: str, tag: str) -> None:
    if not DATA_ARCHIVE.exists():
        raise SystemExit(f"Missing data archive: {DATA_ARCHIVE}")
    data_url = _github_release_asset_url(repo, tag, DATA_ARCHIVE.name)
    _rewrite_default_data_url(data_url)
    _run([sys.executable, "-m", "pip", "wheel", "source/sonogym_reconstruction_data", "--no-deps", "-w", "dist"])
    _run([sys.executable, "-m", "pip", "wheel", "source/spinal_surgery", "--no-deps", "-w", "dist"])
    print(f"[RESULT] rebuilt wheels with DEFAULT_DATA_URL={data_url}", flush=True)


def _request(method: str, url: str, token: str, data: bytes | None = None, content_type: str = "application/json") -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if data is not None:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            body = response.read()
            return json.loads(body.decode("utf-8")) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {method} {url} failed: HTTP {exc.code}\n{detail}") from exc


def _create_or_get_release(repo: str, tag: str, title: str, body: str, token: str) -> dict:
    api = f"https://api.github.com/repos/{repo}"
    try:
        return _request("GET", f"{api}/releases/tags/{urllib.parse.quote(tag, safe='')}", token)
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise
    payload = json.dumps(
        {
            "tag_name": tag,
            "name": title,
            "body": body,
            "draft": False,
            "prerelease": False,
        }
    ).encode("utf-8")
    return _request("POST", f"{api}/releases", token, payload)


def _delete_existing_asset(release: dict, name: str, token: str) -> None:
    for asset in release.get("assets", []):
        if asset.get("name") == name:
            _request("DELETE", asset["url"], token)
            return


def _upload_asset(repo: str, release: dict, path: Path, token: str, overwrite: bool) -> str:
    if overwrite:
        _delete_existing_asset(release, path.name, token)
    upload_url = release["upload_url"].split("{", 1)[0]
    url = f"{upload_url}?name={urllib.parse.quote(path.name)}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = path.read_bytes()
    asset = _request("POST", url, token, data=data, content_type=content_type)
    return asset["browser_download_url"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub repository as owner/name.")
    parser.add_argument("--tag", default="sonogym-reconstruction-v0.1.1")
    parser.add_argument("--title", default="SonoGym Reconstruction v0.1.1")
    parser.add_argument("--body", default="SonoGym reconstruction core wheels and hosted data archive.")
    parser.add_argument("--asset", action="append", dest="assets", default=None)
    parser.add_argument(
        "--prepare-wheels",
        action="store_true",
        help="Rewrite the data package URL to the GitHub Release asset URL and rebuild wheels before upload.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("Set GITHUB_TOKEN before running this script.")
    if args.prepare_wheels:
        _prepare_wheels(args.repo, args.tag)
    assets = [Path(path) for path in (args.assets or DEFAULT_ASSETS)]
    missing = [str(path) for path in assets if not path.exists()]
    if missing:
        raise SystemExit("Missing release assets:\n" + "\n".join(missing))

    release = _create_or_get_release(args.repo, args.tag, args.title, args.body, token)
    urls = {}
    for path in assets:
        print(f"[UPLOAD] {path} ({path.stat().st_size} bytes)", flush=True)
        urls[path.name] = _upload_asset(args.repo, release, path, token, args.overwrite)
        print(f"[URL] {urls[path.name]}", flush=True)

    print(json.dumps({"repo": args.repo, "tag": args.tag, "assets": urls}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

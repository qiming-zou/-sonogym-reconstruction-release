"""Minimal SonoGym package containing only robotic ultrasound reconstruction."""

import os

ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
DEFAULT_ASSETS_DATA_DIR = os.path.abspath(os.path.join(ASSETS_EXT_DIR, "data"))


def _resolve_assets_data_dir() -> str:
    if os.environ.get("SONOGYM_ASSETS_DATA_DIR"):
        return os.path.abspath(os.environ["SONOGYM_ASSETS_DATA_DIR"])
    try:
        from sonogym_reconstruction_data import assets_data_dir
    except ModuleNotFoundError as exc:
        if exc.name != "sonogym_reconstruction_data":
            raise
        return DEFAULT_ASSETS_DATA_DIR

    data_dir = assets_data_dir()
    if data_dir and os.path.exists(data_dir):
        return os.path.abspath(data_dir)
    return DEFAULT_ASSETS_DATA_DIR


ASSETS_DATA_DIR = _resolve_assets_data_dir()
PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

_TASK_IMPORT_ERROR = None


def _is_deferred_runtime_module(name: str | None) -> bool:
    if not name:
        return False
    return name.startswith(("omni", "isaaclab", "isaacsim"))


def register_tasks():
    """Register SonoGym IsaacLab tasks.

    Call this after Isaac Sim's ``AppLauncher`` has initialized the ``omni``
    modules. Importing ``spinal_surgery`` after ``AppLauncher`` still registers
    tasks automatically through the best-effort call below.
    """

    from . import tasks as registered_tasks

    return registered_tasks


try:
    from .lab import *  # noqa: F401,F403,E402
except ModuleNotFoundError as exc:
    if not _is_deferred_runtime_module(exc.name):
        raise
    _TASK_IMPORT_ERROR = exc

try:
    register_tasks()
except ModuleNotFoundError as exc:
    if not _is_deferred_runtime_module(exc.name):
        raise
    _TASK_IMPORT_ERROR = exc

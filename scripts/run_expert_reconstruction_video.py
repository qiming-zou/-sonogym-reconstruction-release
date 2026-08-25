from __future__ import annotations

import runpy
from pathlib import Path


TARGET = Path(__file__).resolve().parents[1] / "visualization" / "run_expert_reconstruction_video.py"
runpy.run_path(str(TARGET), run_name="__main__")

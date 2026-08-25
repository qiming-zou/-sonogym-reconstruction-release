"""Export an interactive 3D HTML view for target anatomy and predicted prior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajectory_generation.patient_conditioned_prior import load_patient_prior_predictor  # noqa: E402
from trajectory_generation.train_patient_conditioned_prior import (  # noqa: E402
    DEFAULT_ASSET_ROOT,
    DEFAULT_CFG,
    load_target_volume,
    target_label_from_name,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export target/prior 3D visualization as HTML.")
    parser.add_argument("--patient_id", required=True)
    parser.add_argument("--trajectory", default=None, help="Optional trajectory .pt file for coverage and path overlay.")
    parser.add_argument("--trajectory_index", type=int, default=0)
    parser.add_argument("--prior_model", default="artifacts/checkpoints/patient_geometry_prior_l4.pt")
    parser.add_argument("--config", default=str(DEFAULT_CFG))
    parser.add_argument("--asset_root", default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument("--target_label", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--max_target_points", type=int, default=7000)
    parser.add_argument("--max_prior_points", type=int, default=9000)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--plotly_js",
        default=None,
        help="Optional local plotly.min.js path. If omitted, the HTML uses the Plotly CDN.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    stride = int(np.ceil(points.shape[0] / max_points))
    return points[::stride]


def _target_points(target_volume: torch.Tensor, max_points: int) -> np.ndarray:
    points = target_volume.nonzero(as_tuple=False).cpu().numpy().astype(np.float32)
    return _downsample_points(points, max_points)


def _prior_points(prior_volume: np.ndarray, threshold: float, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    flat = prior_volume.reshape(-1)
    indices = np.flatnonzero(flat >= float(threshold))
    if indices.size == 0:
        indices = np.argpartition(flat, -min(max_points, flat.size))[-min(max_points, flat.size) :]
    elif indices.size > max_points:
        values = flat[indices]
        top_indices = np.argpartition(values, -max_points)[-max_points:]
        indices = indices[top_indices]
    coords = np.column_stack(np.unravel_index(indices, prior_volume.shape)).astype(np.float32)
    probs = flat[indices].astype(np.float32)
    order = np.argsort(probs)
    return coords[order], probs[order]


def _load_trajectory_metrics(path: str | None, trajectory_index: int) -> dict:
    if path is None:
        return {
            "coverage": None,
            "path_points": None,
            "trajectory_length": None,
            "proxy_total": None,
            "gain_steps": None,
        }
    data = torch.load(path, map_location="cpu")
    coverage = None
    if "final_coverage" in data:
        coverage = float(data["final_coverage"].flatten()[trajectory_index].item())
    cmd_state = data.get("cmd_state")
    path_points = cmd_state[trajectory_index].float().numpy() if cmd_state is not None else None
    proxy = data.get("proxy_reward")
    proxy_total = float(proxy[trajectory_index].sum().item()) if proxy is not None else None
    gain_steps = int((proxy[trajectory_index] > 0).sum().item()) if proxy is not None else None
    return {
        "coverage": coverage,
        "path_points": path_points,
        "trajectory_length": int(path_points.shape[0]) if path_points is not None else None,
        "proxy_total": proxy_total,
        "gain_steps": gain_steps,
    }


def _path_to_volume_grid(
    cmd_state: np.ndarray | None,
    center_label_voxels: np.ndarray,
    volume_size: tuple[int, int, int],
    volume_res: float,
    label_res: float,
    target_points: np.ndarray,
) -> np.ndarray:
    if cmd_state is None or cmd_state.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    x = (cmd_state[:, 0] - center_label_voxels[0]) * label_res / volume_res + volume_size[0] / 2.0
    z = (cmd_state[:, 1] - center_label_voxels[2]) * label_res / volume_res + volume_size[2] / 2.0
    if target_points.size:
        y = np.full_like(x, max(0.0, float(target_points[:, 1].min()) - 2.0))
    else:
        y = np.full_like(x, volume_size[1] * 0.5)
    path = np.column_stack([x, y, z]).astype(np.float32)
    return np.clip(path, 0.0, np.asarray(volume_size, dtype=np.float32) - 1.0)


def _format_metric(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _plotly_loader(plotly_js: str | None) -> str:
    if plotly_js:
        source = Path(plotly_js).read_text()
        return f"<script>{source}</script>"
    return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'


def _html(
    patient_id: str,
    target: np.ndarray,
    prior: np.ndarray,
    prior_prob: np.ndarray,
    path: np.ndarray,
    metrics: dict,
    threshold: float,
    plotly_js: str | None,
) -> str:
    payload = {
        "patient_id": patient_id,
        "target": target.tolist(),
        "prior": prior.tolist(),
        "prior_prob": prior_prob.tolist(),
        "path": path.tolist(),
        "coverage": metrics["coverage"],
        "trajectory_length": metrics["trajectory_length"],
        "proxy_total": metrics["proxy_total"],
        "gain_steps": metrics["gain_steps"],
        "threshold": threshold,
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    coverage_text = _format_metric(metrics["coverage"], 4)
    proxy_text = _format_metric(metrics["proxy_total"], 1)
    gain_text = _format_metric(metrics["gain_steps"])
    length_text = _format_metric(metrics["trajectory_length"])
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{patient_id} target vs predicted prior</title>
  {_plotly_loader(plotly_js)}
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f3f7fa;
      color: #1d252c;
    }}
    header {{
      display: flex;
      gap: 18px;
      align-items: stretch;
      padding: 18px 22px;
      background: #ffffff;
      border-bottom: 1px solid #dce5eb;
    }}
    h1 {{
      margin: 0 20px 0 0;
      font-size: 24px;
      line-height: 1.25;
      min-width: 280px;
    }}
    .metric {{
      min-width: 140px;
      padding: 10px 13px;
      border: 1px solid #dce5eb;
      background: #f8fbfd;
      border-radius: 6px;
    }}
    .label {{
      font-size: 12px;
      color: #66737d;
      margin-bottom: 4px;
    }}
    .value {{
      font-size: 22px;
      font-weight: 700;
    }}
    #plot {{
      width: 100vw;
      height: calc(100vh - 118px);
    }}
  </style>
</head>
<body>
  <header>
    <h1>3D真实结构 / 预测先验<br>{patient_id}</h1>
    <div class="metric"><div class="label">轨迹最终覆盖率</div><div class="value">{coverage_text}</div></div>
    <div class="metric"><div class="label">轨迹长度</div><div class="value">{length_text}</div></div>
    <div class="metric"><div class="label">有效增益步数</div><div class="value">{gain_text}</div></div>
    <div class="metric"><div class="label">重建增益 proxy</div><div class="value">{proxy_text}</div></div>
  </header>
  <div id="plot"></div>
  <script>
    const payload = {data_json};
    const unpack = (points, axis) => points.map(p => p[axis]);
    const target = {{
      type: "scatter3d",
      mode: "markers",
      name: "真实目标结构",
      x: unpack(payload.target, 0),
      y: unpack(payload.target, 1),
      z: unpack(payload.target, 2),
      marker: {{size: 2.5, color: "#15a357", opacity: 0.72}}
    }};
    const prior = {{
      type: "scatter3d",
      mode: "markers",
      name: "预测先验",
      x: unpack(payload.prior, 0),
      y: unpack(payload.prior, 1),
      z: unpack(payload.prior, 2),
      marker: {{
        size: 2.3,
        color: payload.prior_prob,
        colorscale: [[0, "#ffd6ec"], [0.55, "#ff63b7"], [1, "#9d005d"]],
        opacity: 0.55,
        colorbar: {{title: "prior prob"}}
      }}
    }};
    const traces = [prior, target];
    if (payload.path.length > 0) {{
      traces.push({{
        type: "scatter3d",
        mode: "lines",
        name: "轨迹",
        x: unpack(payload.path, 0),
        y: unpack(payload.path, 1),
        z: unpack(payload.path, 2),
        line: {{color: "#111111", width: 6}}
      }});
    }}
    const layout = {{
      margin: {{l: 0, r: 0, t: 20, b: 0}},
      legend: {{x: 0.02, y: 0.98, bgcolor: "rgba(255,255,255,0.82)"}},
      scene: {{
        xaxis: {{title: "x voxel", range: [0, 39]}},
        yaxis: {{title: "y voxel", range: [0, 39]}},
        zaxis: {{title: "z voxel", range: [0, 39]}},
        aspectmode: "cube",
        camera: {{eye: {{x: 1.55, y: 1.25, z: 1.05}}}},
        annotations: [{{
          showarrow: false,
          x: 0,
          y: 39,
          z: 39,
          text: "绿=真实目标；粉=预测先验；黑=轨迹；覆盖率=" + "{coverage_text}",
          font: {{size: 13, color: "#222"}}
        }}]
      }}
    }};
    Plotly.newPlot("plot", traces, layout, {{responsive: true, displaylogo: false}});
  </script>
</body>
</html>
"""


def main() -> None:
    args = _parse_args()
    with Path(args.config).open("r") as f:
        cfg = YAML().load(f)
    target_name = str(cfg["reconstruction"]["target_vertebra"])
    target_label = args.target_label if args.target_label is not None else target_label_from_name(target_name)
    target_volume, center, _label_shape = load_target_volume(
        args.patient_id,
        cfg,
        Path(args.asset_root),
        int(target_label),
    )
    volume_size = tuple(int(value) for value in cfg["reconstruction"]["volume_size"])
    volume_res = float(cfg["reconstruction"]["volume_res"])
    label_res = float(cfg["patient"]["label_res"])

    predictor = load_patient_prior_predictor(args.prior_model, args.device)
    sparse = torch.zeros((1, *target_volume.shape), dtype=torch.float32, device=args.device)
    with torch.inference_mode():
        prior_volume = predictor.predict(sparse, [args.patient_id])[0].detach().cpu().numpy()

    target_points = _target_points(target_volume, args.max_target_points)
    prior_points, prior_prob = _prior_points(prior_volume, args.threshold, args.max_prior_points)
    metrics = _load_trajectory_metrics(args.trajectory, args.trajectory_index)
    path_points = _path_to_volume_grid(
        metrics["path_points"],
        np.asarray(center, dtype=np.float32),
        volume_size,
        volume_res,
        label_res,
        target_points,
    )

    output = Path(args.output or f"artifacts/html/prior_3d_{args.patient_id}.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _html(
            args.patient_id,
            target_points,
            prior_points,
            prior_prob,
            path_points,
            metrics,
            args.threshold,
            args.plotly_js,
        )
    )
    print(f"[RESULT] html={output.resolve()}")
    print(f"[RESULT] patient={args.patient_id} coverage={_format_metric(metrics['coverage'], 4)}")
    print(f"[RESULT] target_points={target_points.shape[0]} prior_points={prior_points.shape[0]}")


if __name__ == "__main__":
    main()

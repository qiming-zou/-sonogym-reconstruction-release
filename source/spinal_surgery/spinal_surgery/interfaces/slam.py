"""Abstract interface for active reconstruction / SLAM planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from .prior import PriorEstimate, PriorExtractor


@dataclass
class ReconstructionState:
    """State snapshot passed to external prior and SLAM algorithms."""

    step: int
    cur_cmd_state: torch.Tensor
    task_env: Any | None = None
    info: Mapping[str, Any] | None = None
    human_rec_volume: torch.Tensor | None = None
    patient_ids: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task_env(
        cls,
        task_env: Any,
        info: Mapping[str, Any],
        step: int,
    ) -> "ReconstructionState":
        rec = getattr(task_env, "surface_reconstructor", None)
        human_rec_volume = None if rec is None else getattr(rec, "human_rec_volume", None)
        patient_ids = None
        if rec is not None and hasattr(rec, "human_list") and hasattr(rec, "env_to_human_inds"):
            import os

            human_ids = [os.path.basename(path.rstrip("/")) for path in rec.human_list]
            env_to_human = rec.env_to_human_inds.detach().cpu().tolist()
            patient_ids = [human_ids[int(index)] for index in env_to_human]
        return cls(
            step=int(step),
            cur_cmd_state=info["cur_cmd_state"],
            task_env=task_env,
            info=info,
            human_rec_volume=human_rec_volume,
            patient_ids=patient_ids,
        )


@dataclass
class GoalCommand:
    """Goal produced by a SLAM planner.

    ``pose`` should match the existing SonoGym command convention:

    - ``(B, 3)``: ``x, z, yaw``
    - ``(B, 4)``: ``x, z, yaw, roll``
    """

    pose: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class SLAMPlanner(ABC):
    """Base class for external active reconstruction / SLAM planners."""

    name = "slam_planner"

    def __init__(self, prior_extractor: PriorExtractor | None = None) -> None:
        self.task_env: Any | None = None
        self.prior_extractor = prior_extractor

    def setup(self, task_env: Any) -> None:
        """Attach the planner to a SonoGym task environment."""

        self.task_env = task_env
        if self.prior_extractor is not None:
            self.prior_extractor.setup(task_env)

    def reset(self) -> None:
        """Reset per-episode planner state."""

        if self.prior_extractor is not None:
            self.prior_extractor.reset()

    def prior(self, state: ReconstructionState) -> PriorEstimate | None:
        """Return the planner's current prior estimate, if any."""

        if self.prior_extractor is None:
            return None
        return self.prior_extractor.update(state)

    @abstractmethod
    def plan(self, state: ReconstructionState, prior: PriorEstimate | None = None) -> GoalCommand:
        """Compute the next command goal."""

    def goal(self, cur_cmd_state: torch.Tensor, step: int, info: Mapping[str, Any] | None = None) -> torch.Tensor:
        """Compatibility wrapper for existing SonoGym rollout loops."""

        if self.task_env is None:
            raise RuntimeError("Call `setup(task_env)` before using `goal(...)`.")
        if info is None:
            info = {"cur_cmd_state": cur_cmd_state}
        state = ReconstructionState.from_task_env(self.task_env, info, step)
        state.cur_cmd_state = cur_cmd_state
        prior = self.prior(state)
        command = self.plan(state, prior)
        return command.pose

    def diagnostics(self) -> Mapping[str, Any]:
        """Return optional implementation-specific diagnostics."""

        return {}

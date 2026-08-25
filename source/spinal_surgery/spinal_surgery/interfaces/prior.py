"""Abstract interface for patient/anatomy prior extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass
class PriorEstimate:
    """A prior estimate produced by a prior extractor.

    Attributes:
        volume: Optional 3D prior with shape ``(B, X, Y, Z)`` or ``(X, Y, Z)``.
        xz: Optional command-space prior with shape ``(B, X, Z)`` or ``(X, Z)``.
        confidence: Scalar or per-environment confidence in ``[0, 1]``.
        metadata: Free-form diagnostics such as registration fitness or model id.
    """

    volume: torch.Tensor | None = None
    xz: torch.Tensor | None = None
    confidence: torch.Tensor | float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "PriorEstimate":
        confidence = self.confidence
        if torch.is_tensor(confidence):
            confidence = confidence.to(device)
        return PriorEstimate(
            volume=None if self.volume is None else self.volume.to(device),
            xz=None if self.xz is None else self.xz.to(device),
            confidence=confidence,
            metadata=dict(self.metadata),
        )


class PriorExtractor(ABC):
    """Base class for external prior extraction algorithms.

    Implementations may be learning-based, registration-based, atlas-based, or
    purely geometric. The core environment only assumes that an extractor can be
    reset and can produce a ``PriorEstimate`` from the current reconstruction
    state.
    """

    name = "prior_extractor"

    def __init__(self) -> None:
        self.task_env: Any | None = None

    def setup(self, task_env: Any) -> None:
        """Attach the extractor to a SonoGym task environment."""

        self.task_env = task_env

    def reset(self) -> None:
        """Reset any per-episode internal state."""

    @abstractmethod
    def update(self, state: Any) -> PriorEstimate:
        """Update and return the latest prior estimate.

        ``state`` is normally a ``ReconstructionState`` from
        ``spinal_surgery.interfaces.slam``. It is typed as ``Any`` here to avoid
        a hard import cycle and to keep simple external implementations light.
        """

    def diagnostics(self) -> Mapping[str, Any]:
        """Return optional implementation-specific diagnostics."""

        return {}

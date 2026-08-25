"""Utilities for anatomy-prior reward and planning."""

from __future__ import annotations

import torch


def load_prior(path: str, device: torch.device | str = "cpu") -> dict:
    prior = torch.load(path, map_location=device)
    if "prior_volume" not in prior:
        raise KeyError(f"Anatomy prior checkpoint {path} does not contain `prior_volume`.")
    return prior


def prior_covered_mass(rec, prior_volume: torch.Tensor) -> torch.Tensor:
    prior = prior_volume.to(rec.human_rec_volume.device)
    return (rec.human_rec_volume.float() * prior.unsqueeze(0)).sum(dim=(1, 2, 3))

"""Image-conditioned offline policy networks for reconstruction trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Normalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.std

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std + self.mean


class ImageEncoder(nn.Module):
    def __init__(self, feature_dim: int, input_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, feature_dim),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(1)
        image = image.float() / 255.0
        return self.net(image)


class TanhGaussianActor(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        feature_dim: int,
        action_limit: torch.Tensor,
        input_channels: int = 1,
    ):
        super().__init__()
        self.encoder = ImageEncoder(feature_dim, input_channels)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("action_limit", action_limit)

    def distribution(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.head(self.encoder(image))
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-5.0, 2.0)
        return mean, log_std

    def sample(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution(image)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        tanh_action = torch.tanh(raw_action)
        action = tanh_action * self.action_limit
        log_prob = dist.log_prob(raw_action) - torch.log(
            self.action_limit * (1.0 - tanh_action.pow(2)) + 1e-6
        )
        return action, log_prob.sum(dim=-1), torch.tanh(mean) * self.action_limit

    def log_prob(self, image: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution(image)
        std = log_std.exp()
        scaled = (action / self.action_limit).clamp(-0.999, 0.999)
        raw_action = 0.5 * (torch.log1p(scaled) - torch.log1p(-scaled))
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(raw_action) - torch.log(self.action_limit * (1.0 - scaled.pow(2)) + 1e-6)
        return log_prob.sum(dim=-1), torch.tanh(mean) * self.action_limit

    def forward(self, image: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            mean, _ = self.distribution(image)
            return torch.tanh(mean) * self.action_limit
        action, _, _ = self.sample(image)
        return action


class ImageCommandActor(nn.Module):
    def __init__(
        self,
        action_dim: int,
        command_dim: int,
        hidden_dim: int,
        feature_dim: int,
        action_limit: torch.Tensor,
        input_channels: int = 1,
    ):
        super().__init__()
        self.encoder = ImageEncoder(feature_dim, input_channels)
        self.head = nn.Sequential(
            nn.Linear(feature_dim + command_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("action_limit", action_limit)

    def distribution(self, image: torch.Tensor, command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.head(torch.cat([self.encoder(image), command.float()], dim=-1))
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-5.0, 2.0)
        return mean, log_std

    def sample(self, image: torch.Tensor, command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution(image, command)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        tanh_action = torch.tanh(raw_action)
        action = tanh_action * self.action_limit
        log_prob = dist.log_prob(raw_action) - torch.log(
            self.action_limit * (1.0 - tanh_action.pow(2)) + 1e-6
        )
        return action, log_prob.sum(dim=-1), torch.tanh(mean) * self.action_limit

    def log_prob(self, image: torch.Tensor, command: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution(image, command)
        std = log_std.exp()
        scaled = (action / self.action_limit).clamp(-0.999, 0.999)
        raw_action = 0.5 * (torch.log1p(scaled) - torch.log1p(-scaled))
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(raw_action) - torch.log(self.action_limit * (1.0 - scaled.pow(2)) + 1e-6)
        return log_prob.sum(dim=-1), torch.tanh(mean) * self.action_limit

    def forward(self, image: torch.Tensor, command: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            mean, _ = self.distribution(image, command)
            return torch.tanh(mean) * self.action_limit
        action, _, _ = self.sample(image, command)
        return action


class ImageCritic(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, feature_dim: int, input_channels: int = 1):
        super().__init__()
        self.encoder = ImageEncoder(feature_dim, input_channels)
        self.q = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = self.encoder(image)
        return self.q(torch.cat([z, action], dim=-1)).squeeze(-1)


class ImageCommandCritic(nn.Module):
    def __init__(
        self,
        action_dim: int,
        command_dim: int,
        hidden_dim: int,
        feature_dim: int,
        input_channels: int = 1,
    ):
        super().__init__()
        self.encoder = ImageEncoder(feature_dim, input_channels)
        self.q = nn.Sequential(
            nn.Linear(feature_dim + command_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor, command: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = self.encoder(image)
        return self.q(torch.cat([z, command.float(), action], dim=-1)).squeeze(-1)


class ImageValue(nn.Module):
    def __init__(self, hidden_dim: int, feature_dim: int, input_channels: int = 1):
        super().__init__()
        self.encoder = ImageEncoder(feature_dim, input_channels)
        self.v = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.v(self.encoder(image)).squeeze(-1)


class ImageCommandValue(nn.Module):
    def __init__(self, command_dim: int, hidden_dim: int, feature_dim: int, input_channels: int = 1):
        super().__init__()
        self.encoder = ImageEncoder(feature_dim, input_channels)
        self.v = nn.Sequential(
            nn.Linear(feature_dim + command_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        return self.v(torch.cat([self.encoder(image), command.float()], dim=-1)).squeeze(-1)

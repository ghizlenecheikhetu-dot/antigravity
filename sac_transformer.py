from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class TransformerFeatureExtractor(nn.Module):
    """Encode a state sequence and return the last timestep representation."""

    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x)
        return x[:, -1, :]


class SACActor(nn.Module):
    """Gaussian policy head used by SAC."""

    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(features)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self(features)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)

        # Tanh-squash log-prob correction
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        squashed_mu = torch.tanh(mu)
        return action, log_prob, squashed_mu


class SACCritic(nn.Module):
    """Double Q critic."""

    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()

        def make_q():
            return nn.Sequential(
                nn.Linear(feature_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([features, actions], dim=-1)
        return self.q1(xu), self.q2(xu)


class SACModel(nn.Module):
    """Convenience wrapper: feature extractor + actor + critic."""

    def __init__(self, state_dim: int, action_dim: int, d_model: int = 128):
        super().__init__()
        self.feature_extractor = TransformerFeatureExtractor(input_dim=state_dim, d_model=d_model)
        self.actor = SACActor(feature_dim=d_model, action_dim=action_dim)
        self.critic = SACCritic(feature_dim=d_model, action_dim=action_dim)

    def encode(self, state_seq: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(state_seq)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, src_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * src_param.data + (1.0 - tau) * target_param.data)


def run_demo(batch_size: int = 32, seq_len: int = 24, state_dim: int = 6, action_dim: int = 2, seed: int = 7):
    torch.manual_seed(seed)
    model = SACModel(state_dim=state_dim, action_dim=action_dim)
    state_seq = torch.randn(batch_size, seq_len, state_dim)

    features = model.encode(state_seq)
    action, log_prob, _ = model.actor.sample(features)
    q1, q2 = model.critic(features, action)

    print("features:", tuple(features.shape))
    print("action:", tuple(action.shape), "log_prob:", tuple(log_prob.shape))
    print("q1:", tuple(q1.shape), "q2:", tuple(q2.shape))


if __name__ == "__main__":
    run_demo()

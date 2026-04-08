from __future__ import annotations

import copy
import math
import random
from collections import deque

import torch
import torch.nn.functional as F

from charging_station_env import ChargingStationEnv
from sac_transformer import SACModel, soft_update


def make_profiles(hours: int = 48):
    pv_profile, grid_prices = [], []
    for t in range(hours):
        daylight = max(0.0, math.sin(((t % 24) - 6) * math.pi / 12))
        pv_profile.append(55.0 * daylight)
        if 18 <= (t % 24) <= 22:
            grid_prices.append(0.19)
        elif 0 <= (t % 24) <= 5:
            grid_prices.append(0.10)
        else:
            grid_prices.append(0.11)
    return pv_profile, grid_prices


def make_scenario():
    return [
        {"arrival_time": 7, "initial_soc": 0.20, "target_soc": 0.90, "capacity": 60, "duration": 8, "departure_time": 15},
        {"arrival_time": 8, "initial_soc": 0.35, "target_soc": 0.85, "capacity": 75, "duration": 7, "departure_time": 15},
        {"arrival_time": 17, "initial_soc": 0.25, "target_soc": 0.90, "capacity": 70, "duration": 6, "departure_time": 23},
        {"arrival_time": 19, "initial_soc": 0.30, "target_soc": 0.80, "capacity": 82, "duration": 8, "departure_time": 27},
    ]


def encode_obs(obs):
    pv, batt_soc, grid_price, hour, ev_needed = obs
    hour_angle = (hour % 24) / 24.0 * 2.0 * math.pi
    return [
        pv / 60.0,
        batt_soc / 150.0,
        (grid_price - 0.10) / 0.10,
        math.sin(hour_angle),
        math.cos(hour_angle),
        min(ev_needed / 200.0, 5.0),
    ]


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.data = deque(maxlen=capacity)

    def add(self, s_seq, a, r, ns_seq, d):
        self.data.append((s_seq, a, r, ns_seq, d))

    def sample(self, batch_size: int, device):
        batch = random.sample(self.data, batch_size)
        s_seq, a, r, ns_seq, d = zip(*batch)
        s_seq = torch.tensor(s_seq, dtype=torch.float32, device=device)
        a = torch.tensor(a, dtype=torch.float32, device=device)
        r = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
        ns_seq = torch.tensor(ns_seq, dtype=torch.float32, device=device)
        d = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)
        return s_seq, a, r, ns_seq, d

    def __len__(self):
        return len(self.data)


class TransformerSACAgent:
    def __init__(self, state_dim=6, action_dim=2, seq_len=24, device="cpu"):
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.action_dim = action_dim

        self.model = SACModel(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.target_feature = copy.deepcopy(self.model.feature_extractor).to(self.device)
        self.target_critic = copy.deepcopy(self.model.critic).to(self.device)

        self.actor_opt = torch.optim.Adam(self.model.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(
            list(self.model.feature_extractor.parameters()) + list(self.model.critic.parameters()), lr=3e-4
        )
        self.log_alpha = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.005
        self.target_entropy = -float(action_dim)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, state_seq, deterministic=False):
        s = torch.tensor(state_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            feat = self.model.encode(s)
            if deterministic:
                mu, _ = self.model.actor(feat)
                action = torch.tanh(mu)
            else:
                action, _, _ = self.model.actor.sample(feat)
        return action.squeeze(0).cpu().numpy().tolist()

    def update(self, replay: ReplayBuffer, batch_size=64):
        if len(replay) < batch_size:
            return None

        s_seq, a, r, ns_seq, d = replay.sample(batch_size, self.device)

        with torch.no_grad():
            nfeat = self.target_feature(ns_seq)
            next_a, next_logp, _ = self.model.actor.sample(nfeat)
            nq1, nq2 = self.target_critic(nfeat, next_a)
            nmin_q = torch.min(nq1, nq2) - self.alpha.detach() * next_logp
            target_q = r + (1.0 - d) * self.gamma * nmin_q

        feat = self.model.encode(s_seq)
        cq1, cq2 = self.model.critic(feat, a)
        critic_loss = F.mse_loss(cq1, target_q) + F.mse_loss(cq2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        feat_detached = self.model.encode(s_seq).detach()
        new_a, logp, _ = self.model.actor.sample(feat_detached)
        q1_pi, q2_pi = self.model.critic(feat_detached, new_a)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp - min_q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        soft_update(self.target_feature, self.model.feature_extractor, self.tau)
        soft_update(self.target_critic, self.model.critic, self.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
        }


def build_initial_seq(obs_vec, seq_len):
    return [obs_vec[:] for _ in range(seq_len)]


def push_seq(seq, obs_vec):
    seq.append(obs_vec[:])
    if len(seq) > 24:
        seq.pop(0)


def train(num_episodes=10, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    device = "cpu"
    pv_profile, grid_prices = make_profiles(hours=48)
    env = ChargingStationEnv(pv_profile=pv_profile, grid_prices=grid_prices, scenario=make_scenario())
    agent = TransformerSACAgent(seq_len=24, device=device)
    replay = ReplayBuffer(capacity=20000)

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        obs_vec = encode_obs(obs)
        seq = build_initial_seq(obs_vec, 24)

        done = False
        ep_reward = 0.0
        updates = 0

        while not done:
            action = agent.act(seq, deterministic=False)
            action[0] = max(-1.0, min(1.0, action[0]))
            action[1] = max(0.0, min(1.0, (action[1] + 1.0) / 2.0))

            next_obs, reward, done, _trunc, info = env.step(action)
            next_obs_vec = encode_obs(next_obs)
            next_seq = seq[1:] + [next_obs_vec]

            replay.add(seq, action, reward, next_seq, float(done))
            metrics = agent.update(replay, batch_size=64)
            if metrics is not None:
                updates += 1

            seq = next_seq
            ep_reward += reward

        print(
            f"Episode {ep:02d} | reward={ep_reward:.2f} | cost={info['cumulative_cost']:.2f} | "
            f"sat={info.get('satisfaction', 0):.2f}% | pv_usage={info.get('pv_usage_pct', 0):.2f}% | updates={updates}"
        )


if __name__ == "__main__":
    train(num_episodes=10)

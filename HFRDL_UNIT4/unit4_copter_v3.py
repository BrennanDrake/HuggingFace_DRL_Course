#!/usr/bin/env python3
import os
from collections import deque

import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym (Pixelcopter)
import gym
import gym_pygame  # registers Pixelcopter-PLE-v0


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Policy network (must match your saved weights)
# -----------------------------
class Policy(nn.Module):
    def __init__(self, s_size: int, a_size: int, h_size: int):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size * 2)
        self.fc3 = nn.Linear(h_size * 2, a_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state: np.ndarray):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state_t).cpu()
        m = Categorical(probs)
        action = m.sample()
        return int(action.item()), m.log_prob(action)


# -----------------------------
# REINFORCE training loop
# -----------------------------
def reinforce(env, policy, optimizer, n_training_episodes: int, max_t: int, gamma: float, print_every: int):
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []

        state = env.reset()
        # Some gym versions return (obs, info)
        if isinstance(state, tuple):
            state = state[0]

        for _t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)

            step_out = env.step(action)
            # gym: (obs, reward, done, info)
            # gymnasium: (obs, reward, terminated, truncated, info)
            if len(step_out) == 4:
                state, reward, done, _info = step_out
            else:
                state, reward, terminated, truncated, _info = step_out
                done = terminated or truncated

            rewards.append(float(reward))

            if done:
                break

        episode_score = sum(rewards)
        scores_deque.append(episode_score)
        scores.append(episode_score)

        # Compute discounted returns (G_t) backwards
        returns = deque(maxlen=max_t)
        disc_return_t = 0.0
        for r in reversed(rewards):
            disc_return_t = r + gamma * disc_return_t
            returns.appendleft(disc_return_t)

        # Normalize returns for stability
        returns_t = torch.tensor(list(returns), dtype=torch.float32)
        eps = np.finfo(np.float32).eps.item()
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + eps)

        # Policy gradient loss
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns_t):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(f"Episode {i_episode}\tAverage Score (last 100): {np.mean(scores_deque):.2f}")

    return scores


# -----------------------------
# Main
# -----------------------------
def main():
    env_id = "Pixelcopter-PLE-v0"
    env = gym.make(env_id)

    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    # Hyperparameters (training only)
    hparams = {
        "h_size": 64,
        "n_training_episodes": 20000,
        "max_t": 500,
        "gamma": 0.925,
        "lr": 1e-4,
        "print_every": 100,
    }

    print(f"Env: {env_id}")
    print(f"Device: {device}")
    print(f"Obs dim: {s_size}, Actions: {a_size}")
    print(f"Hyperparameters: {hparams}")

    policy = Policy(s_size, a_size, hparams["h_size"]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=hparams["lr"])

    reinforce(
        env=env,
        policy=policy,
        optimizer=optimizer,
        n_training_episodes=hparams["n_training_episodes"],
        max_t=hparams["max_t"],
        gamma=hparams["gamma"],
        print_every=hparams["print_every"],
    )

    os.makedirs("models", exist_ok=True)
    out_path = "models/pixelcopter_policy_state_dict.pt"
    torch.save(policy.state_dict(), out_path)
    print(f"Saved model weights to: {out_path}")

    env.close()


if __name__ == "__main__":
    main()

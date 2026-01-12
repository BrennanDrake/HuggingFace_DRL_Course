#!/usr/bin/env python3
import os
import json
import datetime
import tempfile
from pathlib import Path
from collections import deque

import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym (Unit 4 Pixelcopter)
import gym
import gym_pygame  # required for Pixelcopter-PLE-v0 env registration

# HF Hub
from huggingface_hub import HfApi
from huggingface_hub.repocard import metadata_eval_result, metadata_save

import imageio


# -----------------------------
# Global device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Small helpers for API drift
# -----------------------------
def _unwrap_reset(reset_out):
    # gym: obs
    # gymnasium: (obs, info)
    return reset_out[0] if isinstance(reset_out, tuple) else reset_out


def _unwrap_step(step_out):
    # gym: (obs, reward, done, info)
    # gymnasium: (obs, reward, terminated, truncated, info)
    if len(step_out) == 4:
        obs, reward, done, info = step_out
        return obs, reward, done, info
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        return obs, reward, (terminated or truncated), info
    raise ValueError(f"Unexpected env.step output length: {len(step_out)}")


# -----------------------------
# Policy network
# -----------------------------
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size * 2)
        self.fc3 = nn.Linear(h_size * 2, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state_t).cpu()
        m = Categorical(probs)
        action = m.sample()
        return int(action.item()), m.log_prob(action)


# -----------------------------
# Evaluation + video
# -----------------------------
def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    episode_rewards = []
    for _ in range(n_eval_episodes):
        state = _unwrap_reset(env.reset())
        done = False
        total = 0.0
        t = 0
        while not done and t < max_steps:
            action, _ = policy.act(state)
            step_out = env.step(action)
            state, reward, done, _info = _unwrap_step(step_out)
            total += float(reward)
            t += 1
        episode_rewards.append(total)

    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


def record_video(env, policy, out_path, fps=30, max_steps=10_000):
    images = []

    state = _unwrap_reset(env.reset())
    done = False
    t = 0

    # Some envs render immediately; if not, we'll start collecting after first step
    try:
        img = env.render(mode="rgb_array")
    except TypeError:
        img = env.render()
    if img is not None:
        images.append(img)

    while not done and t < max_steps:
        action, _ = policy.act(state)
        step_out = env.step(action)
        state, _reward, done, _info = _unwrap_step(step_out)

        try:
            img = env.render(mode="rgb_array")
        except TypeError:
            img = env.render()
        if img is not None:
            images.append(img)

        t += 1

    # Write MP4 (imageio chooses codec; ffmpeg system package helps)
    imageio.mimsave(out_path, [np.asarray(im) for im in images], fps=fps)


# -----------------------------
# Push-to-hub pipeline
# -----------------------------
def push_to_hub(repo_id, model, hyperparameters, eval_env, video_fps=30):
    _, repo_name = repo_id.split("/")
    api = HfApi()

    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_directory = Path(tmpdirname)

        # Save weights (portable)
        torch.save(model.state_dict(), local_directory / "model_state_dict.pt")

        # Save hyperparameters
        with open(local_directory / "hyperparameters.json", "w") as f:
            json.dump(hyperparameters, f, indent=2)

        # Evaluate
        mean_reward, std_reward = evaluate_agent(
            eval_env,
            hyperparameters["max_t"],
            hyperparameters["n_evaluation_episodes"],
            model,
        )

        eval_datetime = datetime.datetime.now().isoformat()
        results = {
            "env_id": hyperparameters["env_id"],
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_evaluation_episodes": hyperparameters["n_evaluation_episodes"],
            "eval_datetime": eval_datetime,
        }
        with open(local_directory / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Model card metadata
        env_name = hyperparameters["env_id"]
        metadata = {
            "tags": [
                env_name,
                "reinforce",
                "reinforcement-learning",
                "custom-implementation",
                "deep-rl-course",
            ]
        }

        eval_meta = metadata_eval_result(
            model_pretty_name=repo_name,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=env_name,
            dataset_id=env_name,
        )
        metadata = {**metadata, **eval_meta}

        model_card = f"""\
# Reinforce Agent playing {env_name}

This is a trained **REINFORCE** (policy-gradient) agent playing **{env_name}**.

Trained following Unit 4 of the Hugging Face Deep Reinforcement Learning Course.
"""

        readme_path = local_directory / "README.md"
        with readme_path.open("w", encoding="utf-8") as f:
            f.write(model_card)

        metadata_save(readme_path, metadata)

        # Record video (use eval_env passed in)
        video_path = local_directory / "replay.mp4"
        record_video(eval_env, model, str(video_path), fps=video_fps)

        # Upload
        api.upload_folder(repo_id=repo_id, folder_path=local_directory, path_in_repo=".")
        print(f"Pushed to Hub: {repo_url}")
        print(f"Eval mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")


# -----------------------------
# REINFORCE training
# -----------------------------
def reinforce(env, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []

        state = _unwrap_reset(env.reset())
        for _t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)

            step_out = env.step(action)
            state, reward, done, _info = _unwrap_step(step_out)
            rewards.append(float(reward))

            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # discounted returns
        returns = deque(maxlen=max_t)
        disc_return_t = 0.0
        for r in reversed(rewards):
            disc_return_t = r + gamma * disc_return_t
            returns.appendleft(disc_return_t)

        returns = torch.tensor(list(returns), dtype=torch.float32)
        eps = np.finfo(np.float32).eps.item()
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}")

    return scores


# -----------------------------
# Main
# -----------------------------
def main():
    env_id = "Pixelcopter-PLE-v0"

    # For video to work reliably, some envs want render_mode configured at make-time.
    # If this env doesn't support render_mode, the record_video() fallbacks handle it.
    env = gym.make(env_id)
    eval_env = gym.make(env_id)

    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    hparams = {
        "h_size": 64,
        "n_training_episodes": 20000,
        "n_evaluation_episodes": 200,  # keep reasonable locally; raise if you want
        "max_t": 500,
        "gamma": 0.925,
        "lr": 1e-4,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size,
    }

    policy = Policy(s_size, a_size, hparams["h_size"]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=hparams["lr"])

    print(f"Using device: {device}")
    reinforce(
        env=env,
        policy=policy,
        optimizer=optimizer,
        n_training_episodes=hparams["n_training_episodes"],
        max_t=hparams["max_t"],
        gamma=hparams["gamma"],
        print_every=100,
    )

    os.makedirs("models", exist_ok=True)
    torch.save(policy.state_dict(), "models/pixelcopter_policy_state_dict.pt")
    print("Saved local weights to models/pixelcopter_policy_state_dict.pt")

    # Optional: push to hub (requires you to be logged in)
    # Best practice locally: run `huggingface-cli login` once in terminal.
    repo_id = "BrennanDrake/PixelCopter-v1"
    push_to_hub(
        repo_id=repo_id,
        model=policy,
        hyperparameters=hparams,
        eval_env=eval_env,
        video_fps=30,
    )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

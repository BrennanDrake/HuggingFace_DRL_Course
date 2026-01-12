#!/usr/bin/env python3
import os
import json
import datetime
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gym
import gym_pygame  # registers Pixelcopter-PLE-v0

import imageio

from huggingface_hub import HfApi
from huggingface_hub.repocard import metadata_eval_result, metadata_save


# -----------------------------
# Config (edit these)
# -----------------------------
ENV_ID = "Pixelcopter-PLE-v0"
MODEL_PATH = "models/pixelcopter_policy_state_dict.pt"  # your local weights
H_SIZE = 64  # MUST match the trained model

N_EVAL_EPISODES = 200
MAX_STEPS = 500
EVAL_SEED = 42

VIDEO_FPS = 30
VIDEO_MAX_STEPS = 1000  # cap to keep file size reasonable

# Use a separate repo to avoid confusion while iterating
REPO_ID = "BrennanDrake/PixelCopter-v1"


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Policy (must match training)
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
# Eval + video helpers
# -----------------------------
def _make_env(env_id):
    return gym.make(env_id)


def load_policy(env_id: str, model_path: str, h_size: int) -> Policy:
    # infer spaces from a temporary env
    tmp = _make_env(env_id)
    s_size = tmp.observation_space.shape[0]
    a_size = tmp.action_space.n
    tmp.close()

    policy = Policy(s_size, a_size, h_size).to(device)
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def evaluate_fresh_env(env_id: str, policy: Policy, n_eval_episodes: int, max_steps: int, seed: int):
    rewards = []
    for ep in range(n_eval_episodes):
        env = _make_env(env_id)
        try:
            state = env.reset(seed=seed + ep)
        except TypeError:
            state = env.reset()

        done = False
        total = 0.0
        for _ in range(max_steps):
            action, _ = policy.act(state)
            state, reward, done, _info = env.step(action)
            total += float(reward)
            if done:
                break

        env.close()
        rewards.append(total)

    return float(np.mean(rewards)), float(np.std(rewards))


def record_video(env_id: str, policy: Policy, out_path: Path, fps: int = 30, max_steps: int = 1000, seed: int = 123):
    env = _make_env(env_id)
    try:
        state = env.reset(seed=seed)
    except TypeError:
        state = env.reset()

    frames = []
    done = False

    # Some envs require render(mode=...) signature
    def _render():
        try:
            return env.render(mode="rgb_array")
        except TypeError:
            return env.render()

    img = _render()
    if img is not None:
        frames.append(img)

    steps = 0
    while not done and steps < max_steps:
        action, _ = policy.act(state)
        state, _reward, done, _info = env.step(action)
        img = _render()
        if img is not None:
            frames.append(img)
        steps += 1

    env.close()

    # Save MP4
    imageio.mimsave(str(out_path), [np.asarray(f) for f in frames], fps=fps)


# -----------------------------
# Push
# -----------------------------
def push_eval_artifacts(repo_id: str, env_id: str, policy: Policy, model_path: str,
                        mean_reward: float, std_reward: float, hparams: dict,
                        video_fps: int):
    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        d = Path(tmpdirname)

        # Save model weights
        d_model = d / "model_state_dict.pt"
        # Keep the exact bytes as your local file (avoids accidental device differences)
        with open(model_path, "rb") as src, open(d_model, "wb") as dst:
            dst.write(src.read())

        # Save hyperparameters
        with open(d / "hyperparameters.json", "w") as f:
            json.dump(hparams, f, indent=2)

        # Save results
        results = {
            "env_id": env_id,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_evaluation_episodes": hparams["n_evaluation_episodes"],
            "max_t": hparams["max_t"],
            "eval_datetime": datetime.datetime.now().isoformat(),
        }
        with open(d / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        # README + metadata (IMPORTANT: metrics_value must be numeric)
        metadata = {
            "tags": [
                env_id,
                "reinforce",
                "reinforcement-learning",
                "custom-implementation",
                "deep-rl-course",
            ]
        }

        eval_meta = metadata_eval_result(
            model_pretty_name=repo_id.split("/")[1],
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=float(mean_reward),  # numeric, not formatted string
            dataset_pretty_name=env_id,
            dataset_id=env_id,
        )
        metadata = {**metadata, **eval_meta}

        readme = f"""\
# REINFORCE Agent playing {env_id}

This repository contains a **REINFORCE** agent trained for **{env_id}** (Hugging Face Deep RL Course, Unit 4).

## Evaluation
- Episodes: {hparams["n_evaluation_episodes"]}
- Max steps/episode: {hparams["max_t"]}
- Mean reward: {mean_reward:.2f}
- Std reward: {std_reward:.2f}

Artifacts:
- `model_state_dict.pt` (PyTorch state_dict)
- `results.json` (machine-readable evaluation)
- `replay.mp4` (sample rollout)
"""

        readme_path = d / "README.md"
        readme_path.write_text(readme, encoding="utf-8")
        metadata_save(readme_path, metadata)

        # Video
        record_video(env_id, policy, d / "replay.mp4", fps=video_fps, max_steps=VIDEO_MAX_STEPS)

        # Upload folder
        api.upload_folder(repo_id=repo_id, folder_path=d, path_in_repo=".")
        print(f"Uploaded eval artifacts to: {repo_url}")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    print(f"Loading policy from: {MODEL_PATH}")
    policy = load_policy(ENV_ID, MODEL_PATH, H_SIZE)

    hparams = {
        "env_id": ENV_ID,
        "h_size": H_SIZE,
        "n_evaluation_episodes": N_EVAL_EPISODES,
        "max_t": MAX_STEPS,
        "seed": EVAL_SEED,
    }

    print(f"Evaluating: env={ENV_ID}, episodes={N_EVAL_EPISODES}, max_steps={MAX_STEPS}, seed={EVAL_SEED}")
    mean_reward, std_reward = evaluate_fresh_env(ENV_ID, policy, N_EVAL_EPISODES, MAX_STEPS, EVAL_SEED)

    print("===================================")
    print(f"Mean reward : {mean_reward:.2f}")
    print(f"Std reward  : {std_reward:.2f}")
    print("===================================")

    print(f"Pushing to repo: {REPO_ID}")
    push_eval_artifacts(
        repo_id=REPO_ID,
        env_id=ENV_ID,
        policy=policy,
        model_path=MODEL_PATH,
        mean_reward=mean_reward,
        std_reward=std_reward,
        hparams=hparams,
        video_fps=VIDEO_FPS,
    )


if __name__ == "__main__":
    main()

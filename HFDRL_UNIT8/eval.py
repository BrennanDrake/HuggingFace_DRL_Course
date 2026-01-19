import argparse
import json
import os
import shutil
import tempfile
import time
from distutils.util import strtobool
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from huggingface_hub import HfApi, upload_folder
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from ppo import Agent, make_env


def generate_metadata(model_name, env_id, mean_reward, std_reward):
    metadata = {
        "tags": [
            env_id,
            "ppo",
            "deep-reinforcement-learning",
            "reinforcement-learning",
            "custom-implementation",
            "deep-rl-course",
        ]
    }
    eval_metadata = metadata_eval_result(
        model_pretty_name=model_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_id,
        dataset_id=env_id,
    )
    return {**metadata, **eval_metadata}


def save_model_card(local_path, model_name, env_id, mean_reward, std_reward, hyperparameters):
    metadata = generate_metadata(model_name, env_id, mean_reward, std_reward)
    params_str = "\n".join(str(vars(hyperparameters)).split(", "))
    model_card = (
        f"# PPO Agent Playing {env_id}\n\n"
        f"This is a trained model of a PPO agent playing {env_id}.\n\n"
        f"# Hyperparameters\n"
        f"```python\n{params_str}\n```\n"
    )
    readme_path = local_path / "README.md"
    readme_path.write_text(model_card, encoding="utf-8")
    metadata_save(readme_path, metadata)


def package_to_hub(
    repo_id,
    model,
    hyperparameters,
    mean_reward,
    std_reward,
    n_eval_episodes,
    video_src_dir,
    commit_message,
    private,
):
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        torch.save(model.state_dict(), tmpdirname / "model.pt")
        results = {
            "env_id": hyperparameters.gym_id,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_evaluation_episodes": n_eval_episodes,
        }
        (tmpdirname / "results.json").write_text(json.dumps(results, indent=2))

        if video_src_dir and video_src_dir.exists():
            videos = sorted(video_src_dir.rglob("*.mp4"))
            if videos:
                shutil.copy2(videos[0], tmpdirname / "replay.mp4")

        save_model_card(
            tmpdirname,
            model_name="PPO",
            env_id=hyperparameters.gym_id,
            mean_reward=mean_reward,
            std_reward=std_reward,
            hyperparameters=hyperparameters,
        )

        upload_folder(
            repo_id=repo_id,
            folder_path=str(tmpdirname),
            path_in_repo="",
            commit_message=commit_message,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="path to a .pt state_dict")
    parser.add_argument("--gym-id", type=str, default="LunarLander-v2", help="the id of the gym environment")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--num-episodes", type=int, default=5, help="number of episodes to run")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="whether to capture videos of the agent performances")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be used when available")
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="use argmax policy instead of sampling")
    parser.add_argument("--push-to-hub", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="upload checkpoint and eval artifacts to the Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default=None,
                        help="Hub repo id, e.g. username/ppo-lunarlander-v3")
    parser.add_argument("--hub-commit-message", type=str, default="Add evaluation artifacts",
                        help="commit message used when pushing to the Hub")
    parser.add_argument("--hub-private", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="create the repo as private if it does not exist")
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = f"{args.gym_id}__eval__{args.seed}__{int(time.time())}"
    os.makedirs("videos", exist_ok=True)

    gym_id = args.gym_id
    if gym_id == "LunarLander-v2":
        print("LunarLander-v2 is deprecated in gymnasium; using LunarLander-v3 for evaluation.")
        gym_id = "LunarLander-v3"

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    envs = gym.vector.SyncVectorEnv(
        [make_env(gym_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        agent.load_state_dict(ckpt["model"])
    else:
        agent.load_state_dict(ckpt)
    agent.eval()

    obs, _ = envs.reset(seed=args.seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    episodes = 0
    ep_return = np.zeros(envs.num_envs, dtype=np.float64)
    ep_length = np.zeros(envs.num_envs, dtype=np.int64)
    episode_returns = []
    episode_lengths = []

    while episodes < args.num_episodes:
        with torch.no_grad():
            if args.deterministic:
                logits = agent.actor(obs)
                action = torch.argmax(logits, dim=-1)
            else:
                action, _, _, _ = agent.get_action_and_value(obs)

        obs, reward, terminations, truncations, _ = envs.step(action.cpu().numpy())
        done = np.logical_or(terminations, truncations)
        ep_return += reward
        ep_length += 1
        obs = torch.tensor(obs, dtype=torch.float32, device=device)

        if done[0]:
            episodes += 1
            episode_returns.append(float(ep_return[0]))
            episode_lengths.append(int(ep_length[0]))
            print(f"episode={episodes}, return={ep_return[0]}, length={ep_length[0]}")
            ep_return[0] = 0.0
            ep_length[0] = 0
        if done[0] and episodes >= args.num_episodes:
            break

    avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
    avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    std_return = float(np.std(episode_returns)) if episode_returns else 0.0
    print(f"avg_return={avg_return}, avg_length={avg_length}")

    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("--repo-id is required when --push-to-hub is enabled")
        package_to_hub(
            repo_id=args.repo_id,
            model=agent,
            hyperparameters=args,
            mean_reward=avg_return,
            std_reward=std_return,
            n_eval_episodes=len(episode_returns),
            video_src_dir=Path("videos") / run_name,
            commit_message=args.hub_commit_message,
            private=args.hub_private,
        )


if __name__ == "__main__":
    main()

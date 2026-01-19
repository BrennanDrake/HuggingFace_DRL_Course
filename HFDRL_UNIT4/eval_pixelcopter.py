import numpy as np
import torch
import gym
import gym_pygame  # required for Pixelcopter registration

from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

# ------------------
# Device
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------
# Policy (must match training exactly)
# ------------------
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
        return int(action.item())

# ------------------
# Evaluation
# ------------------
def evaluate(
    env_id,
    model_path,
    h_size,
    n_eval_episodes=100,
    max_steps=500,
    seed=42,
):
    # Create a temp env just to get spaces
    tmp_env = gym.make(env_id)
    s_size = tmp_env.observation_space.shape[0]
    a_size = tmp_env.action_space.n
    tmp_env.close()

    # Load policy
    policy = Policy(s_size, a_size, h_size).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    episode_rewards = []

    for ep in range(n_eval_episodes):
        env = gym.make(env_id)
        try:
            state = env.reset(seed=seed + ep)
        except TypeError:
            state = env.reset()

        done = False
        total_reward = 0.0

        for _ in range(max_steps):
            action = policy.act(state)
            state, reward, done, _ = env.step(action)
            total_reward += float(reward)
            if done:
                break

        env.close()
        episode_rewards.append(total_reward)

    mean_r = float(np.mean(episode_rewards))
    std_r = float(np.std(episode_rewards))

    print("===================================")
    print(f"Evaluation episodes : {n_eval_episodes}")
    print(f"Mean reward         : {mean_r:.2f}")
    print(f"Std reward          : {std_r:.2f}")
    print("===================================")

    return mean_r, std_r


if __name__ == "__main__":
    ENV_ID = "Pixelcopter-PLE-v0"
    MODEL_PATH = "models/pixelcopter_policy_state_dict.pt"

    evaluate(
        env_id=ENV_ID,
        model_path=MODEL_PATH,
        h_size=64,          # MUST match training
        n_eval_episodes=200, # HF usually ~100–200
        max_steps=500,
        seed=42,
    )

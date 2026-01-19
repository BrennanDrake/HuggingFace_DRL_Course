import os

import gymnasium as gym
import panda_gym

from huggingface_sb3 import load_from_hub, package_to_hub

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from huggingface_hub import notebook_login

# Load the saved statistics
eval_env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3")])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False

model = A2C.load("a2c-PandaReachDense-v3")

package_to_hub(
    model=model,
    model_name=f"a2c",
    model_architecture="A2C",
    env_id="PandaReachDense-v3",
    eval_env=eval_env,
    repo_id=f"BrennanDrake/a2c-panda-reach-dense-1", # Change the username
    commit_message="Initial commit",
)
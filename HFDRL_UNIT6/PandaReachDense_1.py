import os

import gymnasium as gym
import panda_gym

from huggingface_sb3 import load_from_hub, package_to_hub

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from huggingface_hub import notebook_login




def main():

    env_id = "PandaReachDense-v3"

    # Instantiate the environment
    env = gym.make(env_id)

    #Get the state and action space
    s_size = env.observation_space.shape
    a_size = env.action_space.shape

    print("_____OBSERVATION SPACE_____ \n")
    print("The State Space is: ", s_size)
    print("Sample observation", env.observation_space.sample()) # Get a random observation

    print("_____ACTION SPACE_____ \n")
    print("The Action Space is: ", a_size)
    print("Sample action", env.action_space.sample()) # Get a random action

    env = make_vec_env(env_id, n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = A2C("MultiInputPolicy", env, verbose=1)

    model.learn(total_timesteps=1000000)

    # Save the model and  VecNormalize statistics when saving the agent
    model.save("a2c-PandaReachDense-v3")
    env.save("vec_normalize.pkl")

if __name__ == "__main__":
    main()
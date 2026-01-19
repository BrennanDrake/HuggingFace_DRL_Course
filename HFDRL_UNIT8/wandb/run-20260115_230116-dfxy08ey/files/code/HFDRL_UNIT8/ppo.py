import argparse
import os
from distutils.util import strtobool
import time
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import torch
import gymnasium as gym

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="LunarLander-v3",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=str, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=25000,
        help='total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="if toggled, 'torch.backends.cudnn.deterministic'0=False")
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this expoeriment will be tracked with weights and biases')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help='the wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    ## Random Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    ## uses gpu if available and flag is set on runtime
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    video_kwargs = {
    "video_folder": "videos/train",
    "step_trigger": lambda step: step % 1500 == 0,
    "video_length": 200,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    observation = env.reset()
    for _ in range (2000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated 
        if done:
            observation = env.reset()
            print(f"episodic return: {info['episode']['r']}")
    env.close()
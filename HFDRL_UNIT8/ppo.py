import argparse
import os
from distutils.util import strtobool
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers.vector import RecordVideo

import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--run-name', type=str, default=None,
        help='override the run name used for logging and checkpoints')
    parser.add_argument('--gym-id', type=str, default="LunarLander-v3",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=50000000,
        help='total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="if toggled, 'torch.backends.cudnn.deterministic'0=False")
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--cudnn-benchmark', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='enable cudnn benchmark mode for faster kernels (best for fixed shapes)')
    parser.add_argument('--cuda-tf32', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='allow TF32 on matmul/conv for faster GPU math on Ampere+')
    parser.add_argument('--torch-compile', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='compile the model with torch.compile for potential speedups')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this expoeriment will be tracked with weights and biases')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help='the wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--save-model', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='whether to save the trained model at the end of training')
    parser.add_argument('--save-path', type=str, default="models",
        help='directory to save trained model checkpoints')
    parser.add_argument('--save-optimizer', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='save optimizer state in checkpoint files')
    parser.add_argument('--save-every', type=int, default=0,
        help='save a checkpoint every N updates (0 disables)')
    parser.add_argument('--resume-from', type=str, default=None,
        help='path to a checkpoint to resume from')
    parser.add_argument('--resume-optimizer', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='restore optimizer state when resuming from a checkpoint')
    parser.add_argument('--eval-every', type=int, default=0,
        help='run a quick evaluation every N updates (0 disables)')
    parser.add_argument('--eval-episodes', type=int, default=3,
        help='number of episodes for quick evaluation')
    parser.add_argument('--eval-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='use argmax actions during quick evaluation')
    
    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=8,
        help='the number of parallel game environments')
    parser.add_argument('--num-steps', type=int, default=128,
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggle learning rate annealing for policy and value networks')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=4,
        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=4,
        help='the K epochs to update the policy')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggles advantages normalization')
    parser.add_argument('--clip-coef', type=float, default=0.2,
        help='the surrogate clipping coefficient')
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggles whether or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--ent-coef', type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument('--ent-anneal', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='linearly anneal entropy coefficient over training')
    parser.add_argument('--ent-coef-start', type=float, default=None,
        help='starting entropy coefficient when using --ent-anneal (defaults to --ent-coef)')
    parser.add_argument('--ent-coef-end', type=float, default=0.0,
        help='ending entropy coefficient when using --ent-anneal')
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
        help='the target KL divergence threshold')
    parser.add_argument('--center-reward-coef', type=float, default=0.0,
        help='penalize distance from center using the x-position observation (0 disables)')
    parser.add_argument('--center-reward-anneal', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='linearly anneal center reward coefficient over training')
    parser.add_argument('--center-reward-start', type=float, default=None,
        help='starting center reward coefficient when using --center-reward-anneal (defaults to --center-reward-coef)')
    parser.add_argument('--center-reward-end', type=float, default=0.0,
        help='ending center reward coefficient when using --center-reward-anneal')
    parser.add_argument('--time-penalty', type=float, default=0.0,
        help='per-step negative reward to encourage faster landings (0 disables)')
    parser.add_argument('--time-penalty-anneal', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='linearly increase time penalty over training')
    parser.add_argument('--time-penalty-start', type=float, default=None,
        help='starting time penalty when using --time-penalty-anneal (defaults to 0)')
    parser.add_argument('--time-penalty-end', type=float, default=None,
        help='ending time penalty when using --time-penalty-anneal (defaults to --time-penalty)')
    parser.add_argument('--height-penalty', type=float, default=0.0,
        help='penalize altitude using the y-position observation (0 disables)')
    parser.add_argument('--height-penalty-anneal', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='linearly increase height penalty over training')
    parser.add_argument('--height-penalty-start', type=float, default=None,
        help='starting height penalty when using --height-penalty-anneal (defaults to 0)')
    parser.add_argument('--height-penalty-end', type=float, default=None,
        help='ending height penalty when using --height-penalty-anneal (defaults to --height-penalty)')
    parser.add_argument('--goal-distance-penalty', type=float, default=0.0,
        help='penalize distance to goal using x/y position observation (0 disables)')
    parser.add_argument('--goal-distance-anneal', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='linearly anneal goal distance penalty over training')
    parser.add_argument('--goal-distance-start', type=float, default=None,
        help='starting goal distance penalty when using --goal-distance-anneal (defaults to --goal-distance-penalty)')
    parser.add_argument('--goal-distance-end', type=float, default=0.0,
        help='ending goal distance penalty when using --goal-distance-anneal')

    args = parser.parse_args()
    return args

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(gym_id, render_mode="rgb_array")
        else:
            env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            self._layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            self._layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

if __name__ == "__main__":
    args = parse_args()
    run_name = args.run_name or f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    ## uses gpu if available and flag is set on runtime
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # TRY NOT TO MODIFY: seeding
    args.seed = int(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = args.cudnn_benchmark and not args.torch_deterministic
        torch.backends.cuda.matmul.allow_tf32 = args.cuda_tf32
        torch.backends.cudnn.allow_tf32 = args.cuda_tf32
        if args.cuda_tf32:
            torch.set_float32_matmul_precision("high")

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    eval_envs = None
    if args.eval_every and args.eval_every > 0:
        eval_envs = gym.vector.SyncVectorEnv(
            [make_env(args.gym_id, args.seed + 999, 0, False, run_name)]
        )

    agent = Agent(envs).to(device)
    if args.torch_compile:
        agent = torch.compile(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    recent_returns = deque(maxlen=100)
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    num_updates = args.total_timesteps // args.batch_size
    start_update = 1
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            agent.load_state_dict(ckpt["model"])
            if args.resume_optimizer and "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_update = int(ckpt.get("update", 0)) + 1
            global_step = int(ckpt.get("global_step", 0))
        else:
            agent.load_state_dict(ckpt)
    ep_returns = np.zeros(args.num_envs, dtype=np.float64)
    ep_returns_raw = np.zeros(args.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int64)

    for update in range(start_update, num_updates + 1):
        update_episode_returns = []
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        if args.ent_anneal:
            ent_start = args.ent_coef if args.ent_coef_start is None else args.ent_coef_start
            ent_coef_now = ent_start + frac * (args.ent_coef_end - ent_start)
        else:
            ent_coef_now = args.ent_coef
        if args.time_penalty_anneal:
            tp_start = 0.0 if args.time_penalty_start is None else args.time_penalty_start
            tp_end = args.time_penalty if args.time_penalty_end is None else args.time_penalty_end
            time_penalty_now = tp_start + (1.0 - frac) * (tp_end - tp_start)
        else:
            time_penalty_now = args.time_penalty
        if args.center_reward_anneal:
            cr_start = args.center_reward_coef if args.center_reward_start is None else args.center_reward_start
            cr_end = args.center_reward_end
            center_coef_now = cr_start + frac * (cr_end - cr_start)
        else:
            center_coef_now = args.center_reward_coef
        if args.height_penalty_anneal:
            hp_start = 0.0 if args.height_penalty_start is None else args.height_penalty_start
            hp_end = args.height_penalty if args.height_penalty_end is None else args.height_penalty_end
            height_penalty_now = hp_start + (1.0 - frac) * (hp_end - hp_start)
        else:
            height_penalty_now = args.height_penalty
        if args.goal_distance_anneal:
            gd_start = args.goal_distance_penalty if args.goal_distance_start is None else args.goal_distance_start
            gd_end = args.goal_distance_end
            goal_distance_now = gd_start + frac * (gd_end - gd_start)
        else:
            goal_distance_now = args.goal_distance_penalty

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            raw_reward = reward.copy()
            if center_coef_now != 0.0:
                # Penalize distance from center using x-position (obs[0]) for LunarLander-style envs.
                reward = reward - center_coef_now * np.abs(next_obs[:, 0])
            if time_penalty_now != 0.0:
                reward = reward - time_penalty_now
            if height_penalty_now != 0.0:
                # Penalize altitude using y-position (obs[1]) for LunarLander-style envs.
                reward = reward - height_penalty_now * next_obs[:, 1]
            if goal_distance_now != 0.0:
                # Penalize distance to goal using x/y position (obs[0], obs[1]) for LunarLander-style envs.
                dist_to_goal = np.sqrt(next_obs[:, 0] ** 2 + next_obs[:, 1] ** 2)
                reward = reward - goal_distance_now * dist_to_goal
            next_done = np.logical_or(terminations, truncations)
            ep_returns += reward
            ep_returns_raw += raw_reward
            ep_lengths += 1
            if np.any(next_done):
                done_idx = np.where(next_done)[0]
                for idx in done_idx:
                    recent_returns.append(ep_returns[idx])
                    update_episode_returns.append(ep_returns[idx])
                    writer.add_scalar("charts/episodic_return", ep_returns[idx], global_step)
                    writer.add_scalar("charts/episodic_return_raw", ep_returns_raw[idx], global_step)
                    writer.add_scalar("charts/episodic_length", ep_lengths[idx], global_step)
                ep_returns[done_idx] = 0.0
                ep_returns_raw[done_idx] = 0.0
                ep_lengths[done_idx] = 0
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done, dtype=torch.float32, device=device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextreturn = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextreturn = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * nextreturn
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef_now * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # Compute metrics (batch CPU transfer to reduce sync overhead)
        with torch.no_grad():
            clipfrac_mean = torch.stack(clipfracs).mean().item() if clipfracs and len(clipfracs) > 0 else 0.0
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            # Batch all .item() calls together
            v_loss_val = v_loss.item()
            pg_loss_val = pg_loss.item()
            entropy_loss_val = entropy_loss.item()
            old_approx_kl_val = old_approx_kl.item()
            approx_kl_val = approx_kl.item()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss_val, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss_val, global_step)
        writer.add_scalar("losses/entropy", entropy_loss_val, global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl_val, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl_val, global_step)
        writer.add_scalar("losses/clipfrac", clipfrac_mean, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        if update_episode_returns:
            avg_update_return = sum(update_episode_returns) / len(update_episode_returns)
            if recent_returns:
                avg_recent = sum(recent_returns) / len(recent_returns)
                print(
                    f"episodes_finished={len(update_episode_returns)}, "
                    f"avg_return_update={avg_update_return:.2f}, "
                    f"avg_return_{len(recent_returns)}={avg_recent:.2f}"
                )
            else:
                print(
                    f"episodes_finished={len(update_episode_returns)}, "
                    f"avg_return_update={avg_update_return:.2f}"
                )
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.save_every and update % args.save_every == 0:
            os.makedirs(args.save_path, exist_ok=True)
            save_file = os.path.join(args.save_path, f"{run_name}__u{update}.pt")
            torch.save(agent.state_dict(), save_file)
            print(f"Saved checkpoint to {save_file}")
            if args.save_optimizer:
                ckpt_file = os.path.join(args.save_path, f"{run_name}__u{update}.ckpt")
                torch.save(
                    {
                        "model": agent.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                        "update": update,
                    },
                    ckpt_file,
                )
                print(f"Saved resume checkpoint to {ckpt_file}")

        if eval_envs is not None and update % args.eval_every == 0:
            eval_returns = []
            eval_lengths = []
            eval_obs, _ = eval_envs.reset(seed=args.seed + 999)
            eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device=device)
            while len(eval_returns) < args.eval_episodes:
                with torch.no_grad():
                    if args.eval_deterministic:
                        logits = agent.actor(eval_obs)
                        eval_action = torch.argmax(logits, dim=-1)
                    else:
                        eval_action, _, _, _ = agent.get_action_and_value(eval_obs)
                eval_obs, eval_reward, eval_terms, eval_truncs, _ = eval_envs.step(
                    eval_action.cpu().numpy()
                )
                if not eval_returns:
                    eval_returns.append(0.0)
                    eval_lengths.append(0)
                eval_returns[-1] += float(eval_reward[0])
                eval_lengths[-1] += 1
                if eval_terms[0] or eval_truncs[0]:
                    if len(eval_returns) < args.eval_episodes:
                        eval_returns.append(0.0)
                        eval_lengths.append(0)
                eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device=device)
            mean_return = sum(eval_returns) / len(eval_returns)
            mean_length = sum(eval_lengths) / len(eval_lengths)
            print(
                f"eval_update={update}, episodes={len(eval_returns)}, "
                f"mean_return={mean_return:.2f}, mean_length={mean_length:.1f}"
            )

    if args.save_model:
        os.makedirs(args.save_path, exist_ok=True)
        save_file = os.path.join(args.save_path, f"{run_name}.pt")
        torch.save(agent.state_dict(), save_file)
        print(f"Saved model to {save_file}")
        if args.save_optimizer:
            ckpt_file = os.path.join(args.save_path, f"{run_name}.ckpt")
            torch.save(
                {
                    "model": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "update": num_updates,
                },
                ckpt_file,
            )
            print(f"Saved resume checkpoint to {ckpt_file}")

    envs.close()
    if eval_envs is not None:
        eval_envs.close()
    writer.close()

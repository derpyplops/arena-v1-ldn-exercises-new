# %%
import os
import random
import time
import sys
<<<<<<< HEAD
from dataclasses import dataclass
=======
os.chdir(r"C:\Users\calsm\Documents\AI Alignment\ARENA\arena-v1-ldn-exercises-restructured")
sys.path.append(r"C:\Users\calsm\Documents\AI Alignment\ARENA\arena-v1-ldn-exercises-restructured")
from dataclasses import dataclass, field
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)
import re
import numpy as np
import torch
import torch as t
import gym
<<<<<<< HEAD
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym.spaces import Discrete
from einops import rearrange

from utils import make_env, ppo_parse_args
=======
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from gym.spaces import Discrete
from einops import rearrange, repeat
import wandb
import plotly.express as px
from typing import Optional, Any, Tuple, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict

from w4d3_chapter4_ppo.utils import make_env, PPOArgs, arg_help, plot_cartpole_obs_and_dones, set_global_seeds
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
<<<<<<< HEAD
RUNNING_FROM_FILE = "ipykernel_launcher" in os.path.basename(sys.argv[0])
=======

# %%

@dataclass
class Minibatch:
    obs: t.Tensor
    actions: t.Tensor
    logprobs: t.Tensor
    advantages: t.Tensor
    returns: t.Tensor
    values: t.Tensor

class Memory():

    def __init__(self, envs: gym.vector.SyncVectorEnv, args: PPOArgs, device):
        self.envs = envs
        self.args = args
        self.next_obs = None
        self.next_done = None
        self.next_value = None
        self.device = device
        self.global_step = 0
        self.reset()

    def add(self, *data):
        '''Adds an experience to storage. Called during the rollout phase.
        '''
        info, *experiences = data
        self.experiences.append(experiences)
        for item in info:
            if "episode" in item.keys():
                self.episode_lengths.append(item["episode"]["l"])
                self.episode_returns.append(item["episode"]["r"])
                self.add_vars_to_log(
                    episode_length = item["episode"]["l"],
                    episode_return = item["episode"]["r"],
                )
            self.global_step += 1

    def get_minibatches(self) -> List[Minibatch]:
        '''Computes advantages, and returns minibatches to be used in the 
        learning phase.
        '''
        obs, dones, actions, logprobs, rewards, values = [t.stack(arr) for arr in zip(*self.experiences)]
        advantages = compute_advantages(self.next_value, self.next_done, rewards, values, dones, self.device, self.args.gamma, self.args.gae_lambda)
        returns = advantages + values
        return make_minibatches(
            obs, actions, logprobs, advantages, values, returns, self.args.batch_size, self.args.minibatch_size
        )

    def get_progress_bar_description(self) -> Optional[str]:
        '''Creates a progress bar description, if any episodes have terminated. 
        If not, then the bar's description won't change.
        '''
        if self.episode_lengths:
            global_step = self.global_step
            avg_episode_length = np.mean(self.episode_lengths)
            avg_episode_return = np.mean(self.episode_returns)
            return f"{global_step=:<06}, {avg_episode_length=:<3.0f}, {avg_episode_return=:<3.0f}"

    def reset(self) -> None:
        '''Function to be called at the end of each rollout period, to make 
        space for new experiences to be generated.
        '''
        self.experiences = []
        self.vars_to_log = defaultdict(dict)
        self.episode_lengths = []
        self.episode_returns = []
        if self.next_obs is None:
            self.next_obs = torch.tensor(self.envs.reset()).to(self.device)
            self.next_done = torch.zeros(self.envs.num_envs).to(self.device, dtype=t.float)

    def add_vars_to_log(self, **kwargs):
        '''Add variables to storage, for eventual logging (if args.track=True).
        '''
        self.vars_to_log[self.global_step] |= kwargs

    def log(self):
        '''Logs variables to wandb.
        '''
        for step, vars_to_log in self.vars_to_log.items():
            wandb.log(vars_to_log, step=step)

# %%
class PPOScheduler:
    def __init__(self, optimizer, initial_lr: float, end_lr: float, num_updates: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        '''Implement linear learning rate decay so that after num_updates calls to step, the learning rate is end_lr.
        '''
        self.n_step_calls += 1
        frac = self.n_step_calls / self.num_updates
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)

# %%

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        self.obs_shape = envs.single_observation_space.shape
        self.num_obs = np.array(self.obs_shape).item()
        self.num_actions = envs.single_action_space.n
        self.critic = nn.Sequential(
            # nn.Flatten(),
            layer_init(nn.Linear(self.num_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.num_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.num_actions), std=0.01)
        )

<<<<<<< HEAD
=======
    def rollout(self, memory: Memory, args: PPOArgs, envs: gym.vector.SyncVectorEnv) -> None:
        '''Performs the rollout phase, as described in '37 Implementational 
        Details'.
        '''

        device = memory.device

        obs = memory.next_obs
        done = memory.next_done

        for step in range(args.num_steps):

            # Generate the next set of new experiences (one for each env)
            with t.inference_mode():
                value = self.critic(obs).flatten()
                logits = self.actor(obs)
            probs = Categorical(logits=logits)
            action = probs.sample()
            logprob = probs.log_prob(action)
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            reward = t.from_numpy(reward).to(device)

            # (s_t, d_t, a_t, logpi(a_t|s_t), r_t+1, v(s_t))
            memory.add(info, obs, done, action, logprob, reward, value)

            obs = t.from_numpy(next_obs).to(device)
            done = t.from_numpy(next_done).to(device, dtype=t.float)

        # Store last (obs, done, value) tuple, since we need it to compute advantages
        memory.next_obs = obs
        memory.next_done = done
        # Compute advantages, and store them in memory
        with t.inference_mode():
            memory.next_value = self.critic(obs).flatten()

    def learn(self, memory: Memory, args: PPOArgs, optimizer: optim.Adam, scheduler: PPOScheduler) -> None:
        '''Performs the learning phase, as described in '37 Implementational 
        Details'.
        '''

        for _ in range(args.update_epochs):
            minibatches = memory.get_minibatches()
            # Compute loss on each minibatch, and step the optimizer
            for mb in minibatches:
                logits = self.actor(mb.obs)
                probs = Categorical(logits=logits)
                values = self.critic(mb.obs)
                clipped_surrogate_objective = calc_clipped_surrogate_objective(probs, mb.actions, mb.advantages, mb.logprobs, args.clip_coef)
                value_loss = calc_value_function_loss(values, mb.returns, args.vf_coef)
                entropy_bonus = calc_entropy_bonus(probs, args.ent_coef)
                total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus
                optimizer.zero_grad()
                total_objective_function.backward()
                nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)
                optimizer.step()

        # Step the scheduler
        scheduler.step()

        # Get debug variables, for just the most recent minibatch
        if args.track:
            with t.inference_mode():
                newlogprob = probs.log_prob(mb.actions)
                logratio = newlogprob - mb.logprobs
                ratio = logratio.exp()
                approx_kl = (ratio - 1 - logratio).mean().item()
                clipfracs = [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
            memory.add_vars_to_log(
                learning_rate = optimizer.param_groups[0]["lr"],
                value_loss = value_loss.item(),
                clipped_surrogate_objective = clipped_surrogate_objective.item(),
                entropy = entropy_bonus.item(),
                approx_kl = approx_kl,
                clipfrac = np.mean(clipfracs)
            )

# %%

def make_optimizer(agent: Agent, num_updates: int, initial_lr: float, end_lr: float) -> tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.
    '''
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, num_updates)
    return (optimizer, scheduler)

>>>>>>> 048f2ffb9 (make RL changes, and prereqs)
# %%

def compute_advantages(
    next_value: t.Tensor,
    next_done: t.Tensor,
    rewards: t.Tensor,
    values: t.Tensor,
    dones: t.Tensor,
    device: t.device,
    gamma: float,
    gae_lambda: float,
) -> t.Tensor:
<<<<<<< HEAD
    """Compute advantages using Generalized Advantage Estimation.
    next_value: shape (1, env) - represents V(s_{t+1}) which is needed for the last advantage term
=======
    '''Compute advantages using Generalized Advantage Estimation.
    next_value: shape (env,)
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)
    next_done: shape (env,)
    rewards: shape (t, env)
    values: shape (t, env)
    dones: shape (t, env)
    Return: shape (t, env)
<<<<<<< HEAD
    """
    "SOLUTION"
    T = values.shape[0]
    next_values = torch.concat([values[1:], next_value])
    next_dones = torch.concat([dones[1:], next_done.unsqueeze(0)])
    deltas = rewards + gamma * next_values * (1.0 - next_dones) - values

    advantages = deltas.clone().to(device)
=======
    '''
    T = values.shape[0]
    next_values = torch.concat([values[1:], next_value.unsqueeze(0)])
    next_dones = torch.concat([dones[1:], next_done.unsqueeze(0)])
    deltas = rewards + gamma * next_values * (1.0 - next_dones) - values
    advantages = torch.zeros_like(deltas).to(device)
    advantages[-1] = deltas[-1]
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)
    for t in reversed(range(1, T)):
        advantages[t-1] = deltas[t-1] + gamma * gae_lambda * (1.0 - dones[t]) * advantages[t]
    return advantages

# %%

<<<<<<< HEAD
@dataclass
class Minibatch:
    obs: t.Tensor
    logprobs: t.Tensor
    actions: t.Tensor
    advantages: t.Tensor
    returns: t.Tensor
    values: t.Tensor

def minibatch_indexes(batch_size: int, minibatch_size: int) -> list[np.ndarray]:
=======
def minibatch_indexes(batch_size: int, minibatch_size: int) -> List[np.ndarray]:
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)
    '''Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

    Each index should appear exactly once.
    '''
    assert batch_size % minibatch_size == 0
<<<<<<< HEAD
    
=======
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)
    indices = np.random.permutation(batch_size)
    indices = rearrange(indices, "(mb_num mb_size) -> mb_num mb_size", mb_size=minibatch_size)
    return list(indices)

def make_minibatches(
    obs: t.Tensor,
<<<<<<< HEAD
    logprobs: t.Tensor,
    actions: t.Tensor,
    advantages: t.Tensor,
    values: t.Tensor,
    obs_shape: tuple,
    action_shape: tuple,
    batch_size: int,
    minibatch_size: int,
) -> list[Minibatch]:
    '''Flatten the environment and steps dimension into one batch dimension, then shuffle and split into minibatches.'''
    returns = advantages + values

    data = (obs, logprobs, actions, advantages, returns, values)
    shapes = (obs_shape, (), action_shape, (), (), ())
    return [
        Minibatch(*[d.reshape((-1,) + s)[ind] for d, s in zip(data, shapes)])
        for ind in minibatch_indexes(batch_size, minibatch_size)
    ]

# %%

def calc_policy_loss(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, clip_coef: float
) -> t.Tensor:
    '''Return the policy loss, suitable for maximisation with gradient ascent.
=======
    actions: t.Tensor,
    logprobs: t.Tensor,
    advantages: t.Tensor,
    values: t.Tensor,
    returns: t.Tensor,
    batch_size: int,
    minibatch_size: int,
) -> List[Minibatch]:
    '''Flatten the environment and steps dimension into one batch dimension, then shuffle and split into minibatches.'''

    return [
        Minibatch(
            obs.flatten(0, 1)[ind], 
            actions.flatten(0, 1)[ind], 
            logprobs.flatten(0, 1)[ind], 
            advantages.flatten(0, 1)[ind], 
            values.flatten(0, 1)[ind],
            returns.flatten(0, 1)[ind], 
        )
        for ind in minibatch_indexes(batch_size, minibatch_size)
    ]


if MAIN:
    num_envs = 4
    run_name = "test-run"
    envs = gym.vector.SyncVectorEnv(
        [make_env("CartPole-v1", i, i, False, run_name) for i in range(num_envs)]
    )
    args = PPOArgs()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    memory = Memory(envs, args, device)
    agent = Agent(envs).to(device)
    agent.rollout(memory, args, envs)

    obs = t.stack([e[0] for e in memory.experiences])
    done = t.stack([e[1] for e in memory.experiences])
    plot_cartpole_obs_and_dones(obs, done)

    # def write_to_html(fig, filename):
    #     with open(f"{filename}.html", "w") as f:
    #         f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    
    # write_to_html(fig, '090 Trig Loss Ratio.html')

# %%

def calc_clipped_surrogate_objective(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, clip_coef: float
) -> t.Tensor:
    '''Return the clipped surrogate objective, suitable for maximisation with gradient ascent.
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)

    probs: a distribution containing the actor's unnormalized logits of shape (minibatch, num_actions)

    clip_coef: amount of clipping, denoted by epsilon in Eq 7.
<<<<<<< HEAD

    normalize: if true, normalize mb_advantages to have mean 0, variance 1
    '''
    logits_diff = (probs.log_prob(mb_action) - mb_logprobs)
=======
    '''
    logits_diff = probs.log_prob(mb_action) - mb_logprobs
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)

    r_theta = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / mb_advantages.std()

    non_clipped = r_theta * mb_advantages
    clipped = t.clip(r_theta, 1-clip_coef, 1+clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()

# %%

<<<<<<< HEAD
def calc_value_function_loss(critic: nn.Sequential, mb_obs: t.Tensor, mb_returns: t.Tensor, vf_coef: float) -> t.Tensor:
=======
def calc_value_function_loss(values: t.Tensor, mb_returns: t.Tensor, vf_coef: float) -> t.Tensor:
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)
    '''Compute the value function portion of the loss function.

    vf_coef: the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
<<<<<<< HEAD
    critic_prediction = critic(mb_obs)

    return 0.5 * vf_coef * (critic_prediction - mb_returns).pow(2).mean()

# %%

def calc_entropy_loss(probs: Categorical, ent_coef: float):
    '''Return the entropy loss term.
=======
    return 0.5 * vf_coef * (values - mb_returns).pow(2).mean()

# %%

def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    '''Return the entropy bonus term, suitable for gradient ascent.
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)

    ent_coef: the coefficient for the entropy loss, which weights its contribution to the overall loss. Denoted by c_2 in the paper.
    '''
    return ent_coef * probs.entropy().mean()


# %%

<<<<<<< HEAD
class PPOScheduler:
    def __init__(self, optimizer, initial_lr: float, end_lr: float, num_updates: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        '''Implement linear learning rate decay so that after num_updates calls to step, the learning rate is end_lr.'''
        self.n_step_calls += 1
        frac = self.n_step_calls / self.num_updates
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)

def make_optimizer(agent: Agent, num_updates: int, initial_lr: float, end_lr: float) -> tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, num_updates)
    return (optimizer, scheduler)

# %%
@dataclass
class PPOArgs:
    exp_name: str = os.path.basename(globals().get("__file__", "PPO_implementation").rstrip(".py"))
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "PPOCart"
    wandb_entity: str = None
    capture_video: bool = True
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 0.00025
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 512
    minibatch_size: int = 128

def train_ppo(args):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
=======


def train_ppo(args):

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args), # vars is equivalent to args.__dict__
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
<<<<<<< HEAD
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{key}|{value}|" for (key, value) in vars(args).items()]),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
=======
    set_global_seeds(args.seed)
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
<<<<<<< HEAD
    action_shape = envs.single_action_space.shape
    assert action_shape is not None
=======
    assert envs.single_action_space.shape is not None
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"
    agent = Agent(envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    (optimizer, scheduler) = make_optimizer(agent, num_updates, args.learning_rate, 0.0)
<<<<<<< HEAD
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    global_step = 0
    old_approx_kl = 0.0
    approx_kl = 0.0
    value_loss = t.tensor(0.0)
    policy_loss = t.tensor(0.0)
    entropy_loss = t.tensor(0.0)
    clipfracs = []
    info = []
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    if RUNNING_FROM_FILE:
        from tqdm import tqdm
        progress_bar = tqdm(range(num_updates))
        range_object = progress_bar
    else:
        range_object = range(num_updates)
    
    for _ in range_object:
        for i in range(0, args.num_steps):

            global_step += args.num_envs

            "(1) YOUR CODE: Rollout phase (see detail #1)"
            obs[i] = next_obs
            dones[i] = next_done
            
            with t.inference_mode():
                next_values = agent.critic(next_obs).flatten()
                logits = agent.actor(next_obs)
            probs = Categorical(logits=logits)
            action = probs.sample()
            logprob = probs.log_prob(action)
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            
            rewards[i] = t.from_numpy(reward)
            actions[i] = action
            logprobs[i] = logprob
            values[i] = next_values

            next_obs = t.from_numpy(next_obs).to(device)
            next_done = t.from_numpy(done).float().to(device)

            for item in info:
                if "episode" in item.keys():
                    log_string = f"global_step={global_step}, episodic_return={int(item['episode']['r'])}"
                    if RUNNING_FROM_FILE:
                        progress_bar.set_description(log_string)
                    else:
                        print(log_string)
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break
        with t.inference_mode():
            next_value = rearrange(agent.critic(next_obs), "env 1 -> 1 env")
        advantages = compute_advantages(
            next_value, next_done, rewards, values, dones, device, args.gamma, args.gae_lambda
        )
        clipfracs.clear()
        for _ in range(args.update_epochs):
            minibatches = make_minibatches(
                obs,
                logprobs,
                actions,
                advantages,
                values,
                envs.single_observation_space.shape,
                action_shape,
                args.batch_size,
                args.minibatch_size,
            )
            for mb in minibatches:

                "(2) YOUR CODE: compute loss on the minibatch and step the optimizer (not the scheduler). Do detail #11 (global gradient clipping) here using nn.utils.clip_grad_norm_."
                logits = agent.actor(mb.obs)
                probs = Categorical(logits=logits)
                policy_loss = calc_policy_loss(probs, mb.actions, mb.advantages, mb.logprobs, args.clip_coef)
                value_function_loss = calc_value_function_loss(agent.critic, mb.obs, mb.returns, args.vf_coef)
                entropy_loss = calc_entropy_loss(probs, args.ent_coef)
                total_loss = policy_loss - value_function_loss + entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        scheduler.step()
        (y_pred, y_true) = (mb.values.cpu().numpy(), mb.returns.cpu().numpy())
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        with torch.no_grad():
            newlogprob: t.Tensor = probs.log_prob(mb.actions)
            logratio = newlogprob - mb.logprobs
            ratio = logratio.exp()
            old_approx_kl = (-logratio).mean().item()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # if global_step % 10 == 0:
        #     print("steps per second (SPS):", int(global_step / (time.time() - start_time)))

    "If running one of the Probe environments, will test if the learned q-values are\n    sensible after training. Useful for debugging."
=======
    
    "YOUR CODE HERE: initialise your memory object"
    memory = Memory(envs, args, device)

    progress_bar = tqdm(range(num_updates))
    
    for _ in progress_bar:

        "YOUR CODE HERE: perform rollout and learning steps, and optionally log vars"
        agent.rollout(memory, args, envs)
        agent.learn(memory, args, optimizer, scheduler)
        
        if args.track:
            memory.log()
        
        desc = memory.get_progress_bar_description()
        if desc:
            progress_bar.set_description(desc)

        memory.reset()

    # If running one of the Probe environments, test if learned critic values are sensible after training
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)
    obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
    expected_value_for_probes = [[[1.0]], [[-1.0], [+1.0]], [[args.gamma], [1.0]], [[-1.0, 1.0]], [[1.0, -1.0], [-1.0, 1.0]]]
    tolerances = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]
    match = re.match(r"Probe(\d)-v0", args.env_id)
    if match:
        probe_idx = int(match.group(1)) - 1
        obs = t.tensor(obs_for_probes[probe_idx]).to(device)
        value = agent.critic(obs)
        print("Value: ", value)
        expected_value = t.tensor(expected_value_for_probes[probe_idx]).to(device)
        t.testing.assert_close(value, expected_value, atol=tolerances[probe_idx], rtol=0)

    envs.close()
<<<<<<< HEAD
    writer.close()


if MAIN:
    if RUNNING_FROM_FILE:
        filename = globals().get("__file__", "<filename of this script>")
        print(f"Try running this file from the command line instead:\n\tpython {os.path.basename(filename)} --help")
        args = PPOArgs()
    else:
        args = ppo_parse_args()
=======
    if args.track:
        wandb.finish()


# %%

if MAIN:
    args = PPOArgs()
    # args.track = False
    arg_help(args)
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)
    train_ppo(args)

# %%

<<<<<<< HEAD
# %%
=======









>>>>>>> 048f2ffb9 (make RL changes, and prereqs)
from gym.envs.classic_control.cartpole import CartPoleEnv
import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled
import math

class EasyCart(CartPoleEnv):
    def step(self, action):
        (obs, rew, done, info) = super().step(action)

<<<<<<< HEAD
        cart_posn, card_vel, pole_angle, pole_vel = obs

        # First reward: position should be close to the center
        # result: 
        reward_1 = 1 - (cart_posn / 2.5) ** 2

        reward = reward_1
=======
        x, v, theta, omega = obs

        # First reward: angle should be close to zero
        reward_1 = 1 - abs(theta / 0.2095)
        # Second reward: position should be close to the center
        reward_2 = 1 - abs(x / 2.4)

        reward = 0.3 * reward_1 + 0.7 * reward_2
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)

        return (obs, reward, done, info)

gym.envs.registration.register(id="EasyCart-v0", entry_point=EasyCart, max_episode_steps=500)

if MAIN:
<<<<<<< HEAD
    if RUNNING_FROM_FILE:
        filename = globals().get("__file__", "<filename of this script>")
        print(f"Try running this file from the command line instead:\n\tpython {os.path.basename(filename)} --help")
        args = PPOArgs()
        args.env_id = "EasyCart-v0"
    else:
        args = ppo_parse_args()
    train_ppo(args)
# %%





















































# class EasyCart(CartPoleEnv):

#     def step(self, action):
#         obs, rew, done, info = super().step(action)
#         if "SOLUTION":
#             x, v, theta, omega = obs
#             reward = 1 - (x/2.5)**2
#             return obs, reward, done, info


# class SpinCart(CartPoleEnv):

#     def step(self, action):
#         obs, rew, done, info = super().step(action)
#         if "SOLUTION":
#             x, v, theta, omega = obs
#             reward = 0.5*abs(omega) - (x/2.5)**4
#             if abs(x) > self.x_threshold:
#                 done = True
#             else:
#                 done = False 
#             return obs, reward, done, info

# gym.envs.registration.register(id="SpinCart-v0", entry_point=SpinCart, max_episode_steps=500)

# class DanceCart(CartPoleEnv):

#     def __init__(self):
#         super().__init__()

#         # Angle at which to fail the episode
#         self.theta_threshold_radians = 60 * 2 * math.pi / 360
#         self.x_threshold = 2.4

#         # Angle limit set to 2 * theta_threshold_radians so failing observation
#         # is still within bounds.
#         high = np.array(
#             [
#                 self.x_threshold * 2,
#                 np.finfo(np.float32).max,
#                 self.theta_threshold_radians * 2,
#                 np.finfo(np.float32).max,
#             ],
#             dtype=np.float32,
#         )
#         self.observation_space = spaces.Box(-high, high, dtype=np.float32)


#     def step(self, action):
#         obs, rew, done, info = super().step(action)
#         if "SOLUTION":
#             x, v, theta, omega = obs

#             if abs(x) > self.x_threshold:
#                 done = True
#             else:
#                 done = False 

#             rew = 0.1*abs(v) - max(abs(x) - 1, 0)**2
              

#             theta = (theta + math.pi) % (2 * math.pi) - math.pi #wrap angle around

#             return np.array([x, v, theta, omega]), rew, done, info

# gym.envs.registration.register(id="DanceCart-v0", entry_point=DanceCart, max_episode_steps=1000)



if MAIN:
    if "ipykernel_launcher" in os.path.basename(sys.argv[0]):
        filename = globals().get("__file__", "<filename of this script>")
        print(f"Try running this file from the command line instead:\n\tpython {os.path.basename(filename)} --help")
        args = PPOArgs()
    else:
        args = ppo_parse_args()
=======
    args = PPOArgs()
    args.env_id = "EasyCart-v0"
    # args.track = False
    args.gamma = 0.995
    train_ppo(args)

# %%

class SpinCart(CartPoleEnv):

    def step(self, action):
        obs, rew, done, info = super().step(action)
        # YOUR CODE HERE
        x, v, theta, omega = obs
        # Allow for 360-degree rotation
        done = (abs(x) > self.x_threshold)
        # Reward function incentivises fast spinning while staying still & near centre
        rotation_speed_reward = min(1, 0.1*abs(omega))
        stability_penalty = max(1, abs(x/2.5) + abs(v/10))
        reward = rotation_speed_reward - 0.5 * stability_penalty
        return (obs, reward, done, info)


gym.envs.registration.register(id="SpinCart-v0", entry_point=SpinCart, max_episode_steps=500)

if MAIN:
    args = PPOArgs()
    args.env_id = "SpinCart-v0"
>>>>>>> 048f2ffb9 (make RL changes, and prereqs)
    train_ppo(args)

# %%
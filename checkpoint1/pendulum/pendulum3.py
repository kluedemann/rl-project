import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from sac.utils import *
from sac.plots import running_mean


def create_sac(env, hidden_sizes):
    params = SB3_PARAMS
    params["alpha"] = 0.1
    # params["entropy"] = -1
    params["lr"] = 0.
    # params["lr_critic"] = 1e-3
    params["tau"] = 0.995
    hidden_sizes = [128, 128]
    action_bounds = (env.action_space.low, env.action_space.high)
    obs_dim = env.observation_space.shape[0]
    sac = from_dict(hidden_sizes=hidden_sizes, obs_dim=obs_dim, action_bounds=action_bounds, **params)
    return sac

def create_adaptive(env, hidden_sizes):
    params = SB3_PARAMS
    params["alpha"] = 0.1
    # params["entropy"] = -1
    params["lr"] = 1e-3
    # params["lr_critic"] = 1e-3
    params["tau"] = 0.995
    # hidden_sizes = [128, 128]
    action_bounds = (env.action_space.low, env.action_space.high)
    obs_dim = env.observation_space.shape[0]
    sac = from_dict(hidden_sizes=hidden_sizes, obs_dim=obs_dim, action_bounds=action_bounds, **params)
    return sac

def create_hl(env, hidden_sizes):
    params = SB3_PARAMS
    params["alpha"] = 0.1
    # params["entropy"] = -1
    params["lr"] = 0.
    params["lr_critic"] = 1e-3
    params["tau"] = 0.995
    # hidden_sizes = [128, 128]
    action_bounds = (env.action_space.low, env.action_space.high)
    obs_dim = env.observation_space.shape[0]
    sac = hl_sac(hidden_sizes=hidden_sizes, obs_dim=obs_dim, action_bounds=action_bounds, **params)
    return sac

def create_both(env, hidden_sizes):
    params = SB3_PARAMS
    params["alpha"] = 0.1
    # params["entropy"] = -1
    params["lr"] = 1e-3
    params["lr_critic"] = 1e-3
    params["tau"] = 0.995
    # hidden_sizes = [128, 128]
    action_bounds = (env.action_space.low, env.action_space.high)
    obs_dim = env.observation_space.shape[0]
    sac = hl_sac(hidden_sizes=hidden_sizes, obs_dim=obs_dim, action_bounds=action_bounds, **params)
    return sac

def compute_running_means(res, N):
    new_rewards = {k: [] for k in res.keys()}
    for name in res.keys():
        for i, k in enumerate(res[name]):
            new_rewards[name].append(running_mean(k, N))
    return new_rewards

def plot_entropy(all_entropies):
    all_entropies = compute_running_means(all_entropies, 50)
    for name in all_entropies.keys():
        entropy = -np.asarray(all_entropies[name]) / np.log(2)
        median = np.median(entropy, axis=0)
        n_steps = len(median)
        min_val = np.min(entropy, axis=0)
        max_val = np.max(entropy, axis=0)
        x = range(1, n_steps+1)
        plt.plot(x, median, label=name)
        plt.fill_between(x, min_val, max_val, alpha=0.4)
    plt.legend()
    plt.ylabel("Entropy (bits)")
    plt.xlabel("Training Steps")
    plt.title("Pendulum-v1")
    plt.savefig("./entropy.png", dpi=200, bbox_inches="tight")
    plt.clf()

def plot_rewards(all_rewards, time_steps, update_steps):
    all_rewards = compute_running_means(all_rewards, 10)
    for name in all_rewards.keys():
        rewards = np.asarray(all_rewards[name])
        median = np.median(rewards, axis=0)
        n_steps = len(median)
        min_val = np.min(rewards, axis=0)
        max_val = np.max(rewards, axis=0)
        x = np.arange(1, n_steps+1) * (time_steps // update_steps)
        plt.plot(x, median, label=name)
        plt.fill_between(x, min_val, max_val, alpha=0.4)
    plt.legend()
    plt.ylabel("Total Episode Reward")
    plt.xlabel("Training Step")
    plt.title("Pendulum-v1")
    plt.savefig("./rewards.png", dpi=200, bbox_inches="tight")
    plt.clf()

def main():
    agents = {
        "Base SAC": create_sac,
        "Adaptive Entropy": create_adaptive,
        "HL-Gauss": create_hl,
        "Adaptive + HL": create_both,
    }
    all_rewards = {k: [] for k in agents.keys()}
    all_entropies = {k: [] for k in agents.keys()}
    
    env_name = "Pendulum-v1"
    env = gym.make(env_name)

    log_interval = 100           # print avg reward in the interval
    new_episodes = 200 # max training episodes
    max_timesteps = 200
    train_interval = 5
    save_interval = 1000
    hidden_sizes = [64, 64]

    for random_seed in range(5):
        for name, func in agents.items():
            print(f"Model: {name}, Seed: {random_seed}")
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            
            agent = func(env, hidden_sizes)

            losses = []
            rewards = []
            lengths = []
            i_episode = 0

            filepath = f"./results/compare/{name}-{env_name}-{random_seed}"

            warmup_agent(agent, env, 256, max_timesteps)

            losses, rewards, lengths, i_episode, alphas = train_agent(agent, env, i_episode, new_episodes, max_timesteps, filepath, losses, rewards, lengths, log_interval=log_interval, save_interval=save_interval, train_interval=train_interval)
            all_rewards[name].append(rewards)
            all_entropies[name].append(np.asarray(losses)[:,-1])

        with open("./rewards.pkl", 'wb') as out_file:
            pickle.dump(all_rewards, out_file)

        with open("./entropy.pkl", 'wb') as out_file:
            pickle.dump(all_entropies, out_file)

    plot_rewards(all_rewards, max_timesteps, train_interval)

    plot_entropy(all_entropies)

    return all_entropies, all_rewards


main()

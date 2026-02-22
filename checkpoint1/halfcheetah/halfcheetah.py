import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from sac.utils import *
from sac.plots import running_mean


def create_sac(params):
    params["lr"] = 0.
    params["lr_critic"] = 3e-4
    sac = from_dict(**params)
    return sac

def create_adaptive(params):
    params["lr"] = 1e-3
    params["lr_critic"] = 3e-4
    sac = from_dict(**params)
    return sac

def create_hl(params):
    params["lr"] = 0.
    params["lr_critic"] = 1e-3
    sac = hl_sac(**params)
    return sac

def create_both(params):
    params["lr"] = 1e-3
    params["lr_critic"] = 1e-3
    sac = hl_sac(**params)
    return sac

def compute_running_means(res, N):
    new_rewards = {k: [] for k in res.keys()}
    for name in res.keys():
        for i, k in enumerate(res[name]):
            # logprobs = np.asarray(k)[:, -1]
            new_rewards[name].append(running_mean(k, N))
    return new_rewards

def plot_entropy(all_entropies):
    all_entropies = compute_running_means(all_entropies, 50)
    for name, logprobs in all_entropies.items():
        entropy = -np.asarray(logprobs) / np.log(2)
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
    plt.title("HalfCheetah-v5")
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
    plt.title("HalfCheetah-v5")
    plt.savefig("./rewards.png", dpi=200, bbox_inches="tight")
    plt.clf()

def main():
    n_seeds = 3
    agents = {
        "Base SAC": create_sac,
        "Adaptive Entropy": create_adaptive,
        "HL-Gauss": create_hl,
        "Adaptive + HL": create_both,
    }
    all_rewards = {k: [] for k in agents.keys()}
    all_losses = {k: [] for k in agents.keys()}
    
    env_name = "HalfCheetah-v5"
    env = gym.make(env_name)

    # log_interval = 100           # print avg reward in the interval
    # new_episodes = 200 # max training episodes
    # max_timesteps = 1000
    # train_interval = 5
    # save_interval = 1000
    log_interval = 20           # print avg reward in the interval
    new_episodes = 200 # max training episodes
    max_timesteps = 1000   
    save_interval = 1000
    train_interval = 5
    
    hidden_sizes = [128, 128]
    action_bounds = (env.action_space.low, env.action_space.high)
    obs_dim = env.observation_space.shape[0]

    params = SB3_PARAMS
    params["hidden_sizes"] = hidden_sizes
    params["action_bounds"] = action_bounds
    params["obs_dim"] = obs_dim
    params["alpha"] = 0.02

    for i in range(1):
        for random_seed in range(n_seeds):
            for name, func in agents.items():
                print(f"Model: {name}, Seed: {random_seed}")
                torch.manual_seed(random_seed)
                np.random.seed(random_seed)
                
                agent = func(params)
                # print(all_rewards)

                # if i == 0:
                #     losses = []
                #     rewards = []
                # else:
                #     losses = all_losses[name][random_seed]
                #     rewards = all_rewards[name][random_seed]
                
                losses = []
                rewards = []
                lengths = []
                i_episode = len(rewards)

                filepath = f"./results/compare/{name}-{env_name}-{random_seed}"

                warmup_agent(agent, env, 256, max_timesteps)

                losses, rewards, lengths, i_episode, alphas = train_agent(agent, env, i_episode, new_episodes, max_timesteps, filepath, losses, rewards, lengths, log_interval=log_interval, save_interval=save_interval, train_interval=train_interval)
                
                # if i == 0:
                #     all_losses[name].append(losses)
                #     all_rewards[name].append(rewards)
                # else:
                #     all_losses[name][random_seed] = losses
                #     all_rewards[name][random_seed] = rewards
                all_rewards[name].append(rewards)
                all_losses[name].append(np.asarray(losses)[:,-1])

            with open("./rewards.pkl", 'wb') as out_file:
                pickle.dump(all_rewards, out_file)

            with open("./losses.pkl", 'wb') as out_file:
                pickle.dump(all_losses, out_file)

            plot_rewards(all_rewards, max_timesteps, train_interval)

            plot_entropy(all_losses)

    return all_losses, all_rewards


main()

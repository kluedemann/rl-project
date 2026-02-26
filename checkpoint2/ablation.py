import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from sac.utils import *
from sac.plots import running_mean
from sac.trainer import HockeyTrainer, ACTION_BOUNDS, SCALING
from sac.tournament import PooledTournament, RandomAgent
import hockey.hockey_env as h_env


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
    plt.title("Shooting")
    plt.savefig("./entropy.png", dpi=200, bbox_inches="tight")
    plt.clf()

def plot_rewards(all_rewards, train_iter):
    all_rewards = compute_running_means(all_rewards, 30)
    for name in all_rewards.keys():
        rewards = np.asarray(all_rewards[name])
        median = np.median(rewards, axis=0)
        n_steps = len(median)
        min_val = np.min(rewards, axis=0)
        max_val = np.max(rewards, axis=0)
        x = np.arange(1, n_steps+1) * train_iter
        plt.plot(x, median, label=name)
        plt.fill_between(x, min_val, max_val, alpha=0.4)
    plt.legend()
    plt.ylabel("Total Episode Reward")
    plt.xlabel("Training Step")
    plt.title("Shooting")
    plt.savefig("./rewards.png", dpi=200, bbox_inches="tight")
    plt.clf()

def plot_winrate(all_rewards, train_iter):
    all_rewards = compute_running_means(all_rewards, 30)
    for name in all_rewards.keys():
        rewards = np.asarray(all_rewards[name])
        median = np.median(rewards, axis=0)
        n_steps = len(median)
        min_val = np.min(rewards, axis=0)
        max_val = np.max(rewards, axis=0)
        x = np.arange(1, n_steps+1) * train_iter
        plt.plot(x, median, label=name)
        plt.fill_between(x, min_val, max_val, alpha=0.4)
    plt.legend()
    plt.ylabel("Average Winrate")
    plt.xlabel("Training Step")
    plt.title("Shooting")
    plt.savefig("./winrate.png", dpi=200, bbox_inches="tight")
    plt.clf()

def main():
    n_seeds = 5
    agents = {
        "Base SAC": create_sac,
        "Adaptive Entropy": create_adaptive,
        "HL-Gauss": create_hl,
        "Adaptive + HL": create_both,
    }
    all_rewards = {k: [] for k in agents.keys()}
    all_logprobs = {k: [] for k in agents.keys()}
    all_winrates = {k: [] for k in agents.keys()}
    
    
    log_interval = 100           # print avg reward in the interval
    new_episodes = 1000 # max training episodes
    max_timesteps = 250   
    save_interval = 10000
    train_iter = 30
    
    hidden_sizes = [128, 128]

    params = SB3_PARAMS
    params["hidden_sizes"] = hidden_sizes
    params["action_bounds"] = ACTION_BOUNDS
    params["obs_dim"] = len(SCALING)
    params["obs_scale"] = SCALING
    params["alpha"] = 0.01

    for random_seed in range(n_seeds):
        for name, func in agents.items():
            print(f"Model: {name}, Seed: {random_seed}")
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            
            agent = func(params)
            
            trainer = HockeyTrainer(agent, mode=h_env.Mode.NORMAL)
            # opp = h_env.BasicOpponent(weak=True)
            opp = RandomAgent(ACTION_BOUNDS, random_seed)

            tournament = PooledTournament(random_seed)
            tournament.add_agent(opp)

            filepath = f"./comp/{name}-{random_seed}"

            trainer.warmup(256)
            trainer.train_steps(tournament, new_episodes, train_iter, log_interval, max_timesteps)


            all_rewards[name].append(trainer.logs["Rewards"])
            all_logprobs[name].append(trainer.logs["Logprobs"])
            all_winrates[name].append(2*np.asarray(trainer.logs["Scores"]) + 1)

        with open("./rewards.pkl", 'wb') as out_file:
            pickle.dump(all_rewards, out_file)

        with open("./logprobs.pkl", 'wb') as out_file:
            pickle.dump(all_logprobs, out_file)

        with open("./winrates.pkl", 'wb') as out_file:
            pickle.dump(all_winrates, out_file)

        plot_rewards(all_rewards, train_iter)

        plot_entropy(all_logprobs)

        plot_winrate(all_winrates, train_iter)

    return all_logprobs, all_rewards


main()

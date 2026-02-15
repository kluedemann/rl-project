import numpy as np
import matplotlib.pyplot as plt
from sac.utils import load_logs

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def plot_results(filepath):
    logs = load_logs(filepath)
    q_losses = [logs["Q1_loss"], logs["Q2_loss"]]
    plot_q(q_losses)


def plot_q(q_losses, plot_path, N=50):
    plt.plot(q_losses, alpha=0.4, label=["Q1", "Q2"])
    plt.plot(running_mean(q_losses[:,0], N), color="tab:blue", label="Q1 Mean N=50")
    plt.plot(running_mean(q_losses[:,1], N), color="tab:orange", label="Q2 Mean N=50")
    plt.yscale("log")
    plt.ylabel("Q Loss")
    plt.xlabel("Episode")
    plt.legend()
    plt.title("Value Function Losses by Training Episode")
    plt.savefig(f"{plot_path}/Q_loss.png", dpi=200, bbox_inches="tight")
    plt.show()
    # plt.loglog(np.asarray(losses)[:, :2])

def plot_policy_loss(policy_loss, plot_path, N=50):
    plt.plot(policy_loss, alpha=0.4, label="Policy Loss")
    plt.plot(running_mean(policy_loss, N), color="tab:blue", label="Mean N=50")
    
    plt.ylabel("Policy Loss")
    plt.xlabel("Episode")
    plt.legend()
    plt.title("Policy Loss by Training Episode")
    plt.savefig(f"{plot_path}/policy_loss.png", dpi=200, bbox_inches="tight")
    plt.show()

def plot_entropy(logprobs, plot_path, N=50):
    entropy = -logprobs / np.log(2)
    plt.plot(entropy, alpha=0.4, label="Negative Mean Logprobs")
    plt.plot(running_mean(entropy, N), color="tab:blue", label="Mean N=50")
    
    plt.ylabel("Empirical Entropy (bits)")
    plt.xlabel("Episode")
    plt.legend()
    plt.title("Policy Entropy by Training Episode")
    plt.savefig(f"{plot_path}/entropy.png", dpi=200, bbox_inches="tight")
    plt.show()

def plot_rewards(rewards, plot_path, N=50):
    plt.plot(rewards, alpha=0.4, label="Rewards")
    plt.plot(running_mean(rewards, N), color="tab:blue", label="Mean N=50")
    
    plt.ylabel("Total Episode Reward")
    plt.xlabel("Episode")
    plt.legend()
    plt.title("Rewards by Training Episode")
    plt.savefig(f"{plot_path}/rewards.png", dpi=200, bbox_inches="tight")
    plt.show()
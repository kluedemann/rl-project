import torch

from sac.feedforward import Feedforward
from sac.memory import Memory
from sac.SAC import *
from sac.schedules import *
import pickle
from hl_gauss_pytorch import HLGaussLoss
from sac.hlgauss import HLGaussQ


SCALING = np.asarray([ 1.0,  1.0 , 0.5, 4.0, 4.0, 4.0,  
            1.0,  1.0,  0.5, 4.0, 4.0, 4.0,  
            2.0, 2.0, 10.0, 10.0, 4.0 ,4.0])
ACTION_BOUNDS = (-np.ones(4), np.ones(4))


HOCKEY_PARAMS = {
    # NOTE: Only hidden_sizes is relevant if loading agent
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "tau": 1-0.005,
    "batch_size": 256,
    "gamma": 0.99,
    "alpha": 0.01,
    "loss": "MSE",
    "hidden_sizes": [256, 256],
    "obs_dim": len(SCALING),
    "obs_scale": SCALING,
    "action_bounds": ACTION_BOUNDS
}

# Source: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html#parameters
SB3_PARAMS = {
    # "fit_steps": 1,
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "tau": 1-0.005,
    "batch_size": 256,
    "gamma": 0.99,
    "alpha": 0.1,
    "loss": "MSE",
    # "exp_steps": 100
}

# Source: https://spinningup.openai.com/en/latest/algorithms/sac.html#documentation
SPINNING_UP_PARAMS = {
    # "fit_steps": 1,
    "lr_actor": 1e-3,
    "lr_critic": 1e-3,
    "tau": 0.995,
    "batch_size": 100,
    "gamma": 0.99,
    "alpha": 0.2,
    "loss": "MSE",
    # "exp_steps": 10000
}

LOSSES = {
    "MSE": torch.nn.MSELoss()
}


def get_trained_agent(filepath, params=HOCKEY_PARAMS):
    """Create a trained agent from a state file.
    
    Params:
        filepath - path to saved model state; *.pth
        params - agent parameters
    """
    new_agent = from_dict(**params)
    state = torch.load(filepath)
    new_agent.restore_state(state)
    return new_agent


def from_dict(hidden_sizes, lr_critic, lr_actor, loss, tau, alpha, gamma, batch_size, action_bounds, obs_dim=len(SCALING), obs_scale=1., buffer_size=int(1e6), **kwargs):
    """Create new SAC agent.
    
    Params:
        hidden_sizes - hidden layer widths for value function and policy networks
        lr_critic - critic learning rate
        lr_actor - actor learning rate
        loss - critic loss function; accepts (values, targets)
        tau - Polyak update coefficient
        alpha - entropy regularization coefficient
        gamma - discount factor
        batch_size - update batch size
        action_bounds - (low, high) the bounds for the action space of shape (action_dim,)
        obs_dim - dimension of observation space
        obs_scale - scaling factor to divide observations by; float or np.array (state_dim,)
        buffer_size - maximum size of replay buffer
    
    Returns: the initialized SAC agent
    """
    loss_f = LOSSES[loss]
    observation_dim = obs_dim
    action_dim = action_bounds[0].shape[0]
    
    # Initialize first Q function
    Q1_base = Feedforward(observation_dim+action_dim, hidden_sizes, 1)
    Q1_target = Feedforward(observation_dim+action_dim, hidden_sizes, 1)
    Q1_optim = torch.optim.Adam(Q1_base.parameters(),
                                            lr=lr_critic)
    Q1 = QFunction(Q1_base, Q1_target, Q1_optim, loss_f, tau)

    # Initialize second Q function
    Q2_base = Feedforward(observation_dim+action_dim, hidden_sizes, 1)
    Q2_target = Feedforward(observation_dim+action_dim, hidden_sizes, 1)
    Q2_optim = torch.optim.Adam(Q2_base.parameters(), lr=lr_critic)
    Q2 = QFunction(Q2_base, Q2_target, Q2_optim, loss_f, tau)
    
    # Initialize policy
    policy_base = Feedforward(observation_dim, hidden_sizes, 2*action_dim)
    policy_optim = torch.optim.Adam(policy_base.parameters(), lr=lr_actor)
    policy = TanhGaussianPolicy(policy_base, policy_optim, action_bounds)
    
    # alpha_schedule = ConstantSchedule(alpha)
    entropy = -action_dim
    alpha_schedule = AdaptiveSchedule(alpha=alpha, entropy=entropy, **kwargs)

    # Create SAC agent
    buffer = Memory(buffer_size)
    sac = SAC(Q1, Q2, policy, alpha_schedule=alpha_schedule, gamma=gamma, buffer=buffer, batch_size=batch_size, obs_scale=obs_scale)
    return sac


def hl_sac(hidden_sizes, lr_critic, lr_actor, loss, tau, alpha, gamma, batch_size, action_bounds, obs_dim=len(SCALING), obs_scale=1., buffer_size=int(1e6), **kwargs):
    """Create new SAC agent.
    
    Params:
        hidden_sizes - hidden layer widths for value function and policy networks
        lr_critic - critic learning rate
        lr_actor - actor learning rate
        loss - critic loss function; accepts (values, targets)
        tau - Polyak update coefficient
        alpha - entropy regularization coefficient
        gamma - discount factor
        batch_size - update batch size
        action_bounds - (low, high) the bounds for the action space of shape (action_dim,)
        obs_dim - dimension of observation space
        obs_scale - scaling factor to divide observations by; float or np.array (state_dim,)
        buffer_size - maximum size of replay buffer
    
    Returns: the initialized SAC agent
    """
    n_bins = 50
    loss_f = HLGaussLoss(
        min_value = -30,
        max_value = 12.,
        num_bins = n_bins,
        sigma_to_bin_ratio= 0.75,
        clamp_to_range = True # this was added because if any values fall outside of the bins, the loss is 0 with the current logic
    )
    # loss_f = LOSSES[loss]
    observation_dim = obs_dim
    action_dim = action_bounds[0].shape[0]
    
    # Initialize first Q function
    Q1_base = Feedforward(observation_dim+action_dim, hidden_sizes, n_bins)
    Q1_target = Feedforward(observation_dim+action_dim, hidden_sizes, n_bins)
    Q1_optim = torch.optim.Adam(Q1_base.parameters(),
                                            lr=lr_critic)
    Q1 = HLGaussQ(Q1_base, Q1_target, Q1_optim, loss_f, tau)
    
    # Initialize second Q function
    Q2_base = Feedforward(observation_dim+action_dim, hidden_sizes, n_bins)
    Q2_target = Feedforward(observation_dim+action_dim, hidden_sizes, n_bins)
    Q2_optim = torch.optim.Adam(Q2_base.parameters(), lr=lr_critic)
    Q2 = HLGaussQ(Q2_base, Q2_target, Q2_optim, loss_f, tau)
    
    # Initialize policy
    policy_base = Feedforward(observation_dim, hidden_sizes, 2*action_dim)
    policy_optim = torch.optim.Adam(policy_base.parameters(), lr=lr_actor)
    policy = TanhGaussianPolicy(policy_base, policy_optim, action_bounds)
    
    # alpha_schedule = ConstantSchedule(alpha)
    entropy = -action_dim
    alpha_schedule = AdaptiveSchedule(alpha=alpha, entropy=entropy, **kwargs)

    # Create SAC agent
    buffer = Memory(buffer_size)
    sac = SAC(Q1, Q2, policy, alpha_schedule=alpha_schedule, gamma=gamma, buffer=buffer, batch_size=batch_size, obs_scale=obs_scale)
    return sac


def save_logs(filepath, logs):
    """Save training logs.
    
    Params:
        filepath - the base path to file; extension will be added
        logs - (dict) the logs to save
    """
    with open(f"{filepath}-stat.pkl", 'wb') as f:
        pickle.dump(logs, f)


def load_logs(filepath):
    """Load training logs.
    
    Params:
        filepath - the base path to file; extension will be added
    
    Returns: (dict) - the training logs
    """
    with open(f"{filepath}-stat.pkl", 'rb') as f:
        data = pickle.load(f)
    return data


# ********************** OLD INTERFACE ****************************************

def save_statistics(filepath, rewards, lengths, losses):
    with open(f"{filepath}-stat.pkl", 'wb') as f:
        pickle.dump({"rewards" : rewards, "lengths": lengths, "losses": losses}, f)


def load_stats(filepath):
    with open(f"{filepath}-stat.pkl", 'rb') as f:
        data = pickle.load(f)
    return data["rewards"], data["lengths"], data["losses"]


def warmup_agent(agent, env, n_steps, max_timesteps):
    timestep = 0
    while timestep < n_steps:
        ob, _info = env.reset()
        for t in range(max_timesteps):
            done = False
            a = agent.act(ob)
            (ob_new, reward, done, trunc, _info) = env.step(a)
            agent.store_transition((ob, a, reward, ob_new, done))
            ob=ob_new
            timestep += 1
            if done or trunc or t >= n_steps: break


def train_agent(agent, env, i_episode, new_episodes, max_timesteps, filepath, losses, rewards, lengths, log_interval, save_interval, train_interval):
    alphas = []
    for i in range(new_episodes):
        ob, _info = env.reset()
        total_reward=0
        # e_losses = []
        for t in range(max_timesteps):
            done = False
            a = agent.act(ob)
            (ob_new, reward, done, trunc, _info) = env.step(a)
            total_reward+= reward
            agent.store_transition((ob, a, reward, ob_new, done))
            if (t+1) % train_interval == 0: 
                loss = agent.train()
                alphas.append(agent.alpha_schedule.get_alpha())
                # e_losses.append(loss)
                losses.append(loss)
            ob=ob_new
            if done or trunc: break
        i_episode += 1
        # loss = agent.train()
        # losses.append(loss)
        # print(t, np.asarray(e_losses))
        # losses.append(np.mean(e_losses, axis=(0,1)))

        rewards.append(total_reward)
        lengths.append(t+1)

        # save every 500 episodes
        if i_episode % save_interval == 0:
            print("########## Saving a checkpoint... ##########")
            torch.save(agent.state(), f'{filepath}-{i_episode}.pth')
            save_statistics(filepath, rewards, lengths, losses)
    
        # logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))
    
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
    save_statistics(filepath, rewards, lengths, losses)
    return losses, rewards, lengths, i_episode, alphas

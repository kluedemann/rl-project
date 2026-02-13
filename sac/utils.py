import torch

from sac.feedforward import Feedforward
from sac.memory import Memory
from sac.SAC import *
import pickle

SB3_PARAMS = {
    "fit_steps": 1,
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "tau": 1-0.005,
    "batch_size": 256,
    "gamma": 0.99,
    "alpha": 0.1,
    "loss": "MSE",
    # "exp_steps": 100
}

SPINNING_UP_PARAMS = {
    "fit_steps": 1,
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


def from_dict(env, hidden_sizes, lr_critic, lr_actor, loss, tau, alpha, gamma, batch_size, fit_steps, action_bounds):
    loss_f = LOSSES[loss]
    
    observation_dim = env.observation_space.shape[0]
    action_dim = action_bounds[0].shape[0]
    
    Q1_base = Feedforward(observation_dim+action_dim, hidden_sizes, 1)
    Q1_target = Feedforward(observation_dim+action_dim, hidden_sizes, 1)
    Q1_optim = torch.optim.Adam(Q1_base.parameters(),
                                            lr=lr_critic)
    Q1 = QFunction(Q1_base, Q1_target, Q1_optim, loss_f, tau)
    
    Q2_base = Feedforward(observation_dim+action_dim, hidden_sizes, 1)
    Q2_target = Feedforward(observation_dim+action_dim, hidden_sizes, 1)
    Q2_optim = torch.optim.Adam(Q2_base.parameters(), lr=lr_critic)
    Q2 = QFunction(Q2_base, Q2_target, Q2_optim, loss_f, tau)
    
    policy_base = Feedforward(observation_dim, hidden_sizes, 2*action_dim)
    policy_optim = torch.optim.Adam(policy_base.parameters(), lr=lr_actor)
    policy = TanhGaussianPolicy(policy_base, policy_optim, action_bounds)
    buffer = Memory(int(1e6))
    sac = SAC(Q1, Q2, policy, alpha=alpha, gamma=gamma, buffer=buffer, batch_size=batch_size, fit_steps=fit_steps)
    return sac


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
    return losses, rewards, lengths, i_episode
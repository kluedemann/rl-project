from sac.utils import from_dict, load_stats, save_statistics
import numpy as np
import torch
import hockey.hockey_env as h_env

SCALING = np.asarray([ 1.0,  1.0 , 0.5, 4.0, 4.0, 4.0,  
            1.0,  1.0,  0.5, 4.0, 4.0, 4.0,  
            2.0, 2.0, 10.0, 10.0, 4.0 ,4.0])
ACTION_BOUNDS = (-np.ones(4), np.ones(4))

class HockeyTrainer:

    def __init__(self, params, mode=h_env.Mode.NORMAL):
        self.env = h_env.HockeyEnv(mode=mode)
        self.episode = 0
        self.losses = []
        self.rewards = []
        self.lengths = []
        
        self.agent = from_dict(self.env, action_bounds=ACTION_BOUNDS, obs_scale=SCALING, **params)
    
    def reset(self, mode=h_env.Mode.NORMAL):
        self.env.close()
        self.env = h_env.HockeyEnv(mode=mode)

    def load(self, filepath):
        state = torch.load(filepath)
        self.agent.restore_state(state)
        # self.rewards, self.lengths, self.losses = load_stats(filepath)

    def train(self, opponent, new_episodes, train_interval, log_interval=20, max_timesteps=1000):
        for i in range(new_episodes):
            total_reward = 0
            o, info = self.env.reset()
            obs_agent2 = self.env.obs_agent_two()
            # env.render()

            for j in range(max_timesteps):
                a1 = self.agent.act(o)
                a2 = opponent.act(obs_agent2)

                obs, r, d, t , info = self.env.step(np.hstack([a1,a2]))
                self.agent.store_transition((o, a1, 10*info['winner'], obs, d))
                
                if j % train_interval == 0:
                    loss = self.agent.train()
                    self.losses.append(loss)

                total_reward += 10*info['winner']
                o = obs
                obs_agent2 = self.env.obs_agent_two()
                if d or t: break

            self.episode += 1
            
            self.lengths.append(j+1)
            self.rewards.append(total_reward)

            if self.episode % log_interval == 0:
                self.log_results(log_interval)
                # print(i_episode+1, np.mean(rewards[-10:]))
    
    def log_results(self, log_interval):
        avg_reward = np.mean(self.rewards[-log_interval:])
        print(f"{self.episode:6}: Reward: {avg_reward:.3f}")

    def warmup(self, timesteps, max_timesteps=1000):
        curr = 0
        while curr < timesteps:
            o, info = self.env.reset()
            # obs2 = self.env.obs_agent_two()
            
            for _ in range(max_timesteps):
                a1 = self.agent.act(o)
                a2 = np.random.uniform(-1,1,4)

                obs, r, d, t , info = self.env.step(np.hstack([a1,a2]))
                # obs2 = self.env.obs_agent_two()

                self.agent.store_transition((o, a1, 10*info['winner'], obs, d))
                # self.agent.store_transition((obs2, a2, r, obs, d))

                o = obs
                # o2 = obs2

                curr += 1

                if d or t: break

    def evaluate(self, opponent, episodes=5, render=False, max_timesteps=1000):
        rewards = []
        scores = []
        for i in range(episodes):
            o, info = self.env.reset()
            obs_agent2 = self.env.obs_agent_two()
            
            if render:
                self.env.render()
            
            ep_rewards = []
            for _ in range(max_timesteps):
                if render:
                    self.env.render()
                a1 = self.agent.act(o, noise_scale=0.) # np.random.uniform(-1,1,4)
                a2 = opponent.act(obs_agent2)

                obs, r, d, t , info = self.env.step(np.hstack([a1,a2]))
                self.agent.store_transition((o, a1, 10*info['winner'], obs, d))

                o = obs
                obs_agent2 = self.env.obs_agent_two()

                ep_rewards.append(r)
                if d or t: break
            scores.append(info['winner'])
            rewards.append(np.sum(ep_rewards))
        return rewards, np.mean(scores)

    def save_agent(self, filepath):
        torch.save(self.agent.state(), f'{filepath}-{self.episode}.pth')
        save_statistics(filepath, self.rewards, self.lengths, self.losses)
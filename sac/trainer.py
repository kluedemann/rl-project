from sac.utils import from_dict, load_logs, save_logs, hl_sac
import numpy as np
import torch
import hockey.hockey_env as h_env

SCALING = np.asarray([ 1.0,  1.0 , 0.5, 4.0, 4.0, 4.0,  
            1.0,  1.0,  0.5, 4.0, 4.0, 4.0,  
            2.0, 2.0, 10.0, 10.0, 4.0 ,4.0])
ACTION_BOUNDS = (-np.ones(4), np.ones(4))

DEFAULT_REWARD = lambda info : 10*info["winner"] + info["reward_closeness_to_puck"]
SCORE_REWARD = lambda info : 10*info["winner"]

class HockeyTrainer:

    def __init__(self, params, reward_func=DEFAULT_REWARD, mode=h_env.Mode.NORMAL):
        self.env = h_env.HockeyEnv(mode=mode)
        self.episode = 0
        log_keys = ["Q1_loss", "Q2_loss", "Policy_loss", "Logprobs", "Rewards", "Scores", "Lengths"]
        self.logs = {x: [] for x in log_keys}
        self.reward_func = reward_func
        self.agent = self.create_agent(params)
    
    def create_agent(self, params):
        return from_dict(action_bounds=ACTION_BOUNDS, obs_scale=SCALING, **params)
        # return hl_sac(action_bounds=ACTION_BOUNDS, obs_scale=SCALING, **params)

    def log_losses(self, losses):
        loss_keys = ["Q1_loss", "Q2_loss", "Policy_loss", "Logprobs"]
        for k, l in zip(loss_keys, losses):
            self.logs[k].append(l)

    def reset(self, mode=h_env.Mode.NORMAL):
        self.env.close()
        self.env = h_env.HockeyEnv(mode=mode)

    def load(self, filepath, episode):
        state = torch.load(f"{filepath}-{episode}.pth")
        self.agent.restore_state(state)
        self.logs = load_logs(filepath)

    def train(self, tournament, new_episodes, train_interval, log_interval=20, max_timesteps=1000):
        for i in range(new_episodes):
            total_reward = 0
            o, info = self.env.reset()
            o2 = self.env.obs_agent_two()
            # env.render()

            opponent = tournament.get_opponent(self.agent)

            for j in range(max_timesteps):
                a1 = self.agent.act(o)
                a2 = opponent.act(o2)

                obs, r, d, t , info = self.env.step(np.hstack([a1,a2]))
                obs2 = self.env.obs_agent_two()
                info2 = self.env.get_info_agent_two()
                
                r = self.reward_func(info)
                r2 = self.reward_func(info2)

                self.agent.store_transition((o, a1, r, obs, d))
                self.agent.store_transition((o2, a2, r2, obs2, d))
                
                if j % train_interval == 0:
                    loss = self.agent.train()
                    self.log_losses(loss)

                total_reward += r
                o = obs
                o2 = obs2
                if d or t: break

            self.episode += 1
            
            self.logs["Scores"].append(info["winner"])
            self.logs["Lengths"].append(j+1)
            self.logs["Rewards"].append(total_reward)

            if self.episode % log_interval == 0:
                self.log_results(log_interval)
    
    def log_results(self, log_interval):
        avg_reward = np.mean(self.logs["Rewards"][-log_interval:])
        winrate = 0.5* (np.mean(self.logs["Scores"][-log_interval:])+1)
        print(f"{self.episode:6}: Reward: {avg_reward:8.3f} Winrate: {winrate:8.3f}")

    def warmup(self, timesteps, max_timesteps=1000):
        curr = 0
        while curr < timesteps:
            o, info = self.env.reset()
            o2 = self.env.obs_agent_two()
            
            for _ in range(max_timesteps):
                a1 = self.agent.act(o)
                a2 = self.agent.act(o2)

                obs, r, d, t , info = self.env.step(np.hstack([a1,a2]))
                obs2 = self.env.obs_agent_two()
                info2 = self.env.get_info_agent_two()
                
                r = self.reward_func(info)
                r2 = self.reward_func(info2)

                self.agent.store_transition((o, a1, r, obs, d))
                self.agent.store_transition((o2, a2, r2, obs2, d))

                o = obs
                o2 = obs2

                curr += 1

                if d or t: break

    def evaluate(self, opponent, episodes=5, render=False, max_timesteps=1000, noise_scale=0.):
        rewards = []
        scores = []
        for i in range(episodes):
            o, info = self.env.reset()
            o2 = self.env.obs_agent_two()
            
            if render:
                self.env.render()
            
            total_reward = 0
            for _ in range(max_timesteps):
                if render:
                    self.env.render()
                a1 = self.agent.act(o, noise_scale=noise_scale) # np.random.uniform(-1,1,4)
                a2 = opponent.act(o2)

                obs, r, d, t , info = self.env.step(np.hstack([a1,a2]))
                o2 = self.env.obs_agent_two()

                r = self.reward_func(info)
                
                o = obs
            
                total_reward += r
                if d or t: break
            scores.append(info['winner'])
            rewards.append(total_reward)
        return rewards, scores

    def save_agent(self, filepath):
        torch.save(self.agent.state(), f'{filepath}-{self.episode}.pth')
        save_logs(filepath, self.logs)
        # save_statistics(filepath, self.rewards, self.lengths, self.losses)
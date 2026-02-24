import numpy as np
from sac.utils import create_agent, get_trained_agent
import torch


class PooledTournament:

    def __init__(self, seed=0):
        self.agents = []
        self.rng = np.random.default_rng(seed)
    
    def add_agent(self, agent):
        self.agents.append(agent)

    def get_opponent(self, agent=None):
        return self.rng.choice(self.agents)


class LowMemTournament:

    def __init__(self, seed=0, max_size=10):
        self.paths = []
        self.params = []
        self.rng = np.random.default_rng(seed)
        self.max_size = max_size
    
    def add_agent(self, path, params):
        if len(self.paths) == self.max_size:
            self.paths.pop(0)
            self.params.pop(0)
        self.paths.append(path)
        self.params.append(params)

    def get_opponent(self, agent=None):
        ind = self.rng.integers(len(self.paths))
        path = self.paths[ind]
        params = self.params[ind]
        new_agent = get_trained_agent(path, params)
        # agent = create_agent(**params)
        # state = torch.load(path)
        # agent.restore_state(state[2])
        return new_agent


class MixedTournament:

    def __init__(self, tournaments, weights, seed=0):
        self.tournaments = tournaments
        self.weights = weights
        self.rng = np.random.default_rng(seed)

    def get_opponent(self, agent=None):
        tournament = self.rng.choice(self.tournaments, p=self.weights)
        return tournament.get_opponent(agent)


class RandomAgent:

    def __init__(self, action_bounds, random_seed=0):
        self.action_bounds = action_bounds

    def act(self, obs):
        return np.random.uniform(*self.action_bounds)
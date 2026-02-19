import numpy as np

class PooledTournament:

    def __init__(self, seed=0):
        self.agents = []
        self.rng = np.random.default_rng(seed)
    
    def add_agent(self, agent):
        self.agents.append(agent)

    def get_opponent(self, agent=None):
        return self.rng.choice(self.agents)
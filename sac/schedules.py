import abc
import numpy as np
import torch


class AlphaSchedule(abc.ABC):

    @abc.abstractmethod
    def get_alpha(self):
        pass

    @abc.abstractmethod
    def update(self, logprobs):
        pass

    def state(self):
        return []

class ConstantSchedule(AlphaSchedule):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def get_alpha(self):
        return self.alpha
    
    def update(self, logprobs):
        pass


class AdaptiveSchedule(AlphaSchedule):

    def __init__(self, alpha, entropy, **optim_args):
        super().__init__()
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        self.optim = torch.optim.Adam([self.log_alpha], **optim_args)
        self.entropy = torch.tensor(entropy)

    def get_alpha(self):
        return np.exp(self.log_alpha.detach().numpy())
    
    def update(self, logprobs):
        self.optim.zero_grad()
        losses = -torch.exp(self.log_alpha) * (logprobs + self.entropy)
        loss = losses.mean()
        loss.backward()
        self.optim.step()

    def state(self):
        return (self.log_alpha, self.optim.state_dict())

    def restore_state(self, state):
        with torch.no_grad():
            self.log_alpha.copy_(state[0])
        self.optim.load_state_dict(state[1])
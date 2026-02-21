from sac.SAC import QFunction
import torch
from hl_gauss_pytorch import HLGaussLoss


class HLGaussQ(QFunction):
    """Action-value approximation function
    
    Params:
        base - the network used for learning the approximation
        target - the target network; will be overwritten
        optim - the optimizer from torch.optim
        loss - the loss function used
        tau - the Polyak update coefficient

    Interface:
        fit(s, a, t) - perform an update step
        target_value(s, a) - compute the value of the target network
        state - get network parameters
        restore_state(state) - restore from state
        polyak_update() - update target network
    """

    def target_val(self, observations, actions):
        return self.loss(super().target_val(observations, actions))[:,None]
    
    
    def fit(self, observations, actions, targets): # all arguments should be torch tensors
        return super().fit(observations, actions, targets.squeeze())
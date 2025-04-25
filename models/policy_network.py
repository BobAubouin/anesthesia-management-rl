import torch
import torch.nn as nn
from torch.distributions import TransformedDistribution, SigmoidTransform, AffineTransform


class HierarchicalPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # mean of action distribution
        self.mean_layer = nn.Linear(32, action_dim)
        # log-standard-deviation
        self.std_layer = nn.Linear(32, action_dim)

    def forward(self, x):
        x = self.shared_net(x)
        mean = self.mean_layer(x)  # unbounded output
        log_std = self.std_layer(x)
        log_std = torch.clamp(log_std, min=-5.0, max=1.5)  # bornes raisonnables
        std = torch.exp(log_std)

        action_max = 10
        # create squashed normal distribution between 0 and action_max
        base_dist = torch.distributions.Normal(mean, std)
        # Apply transformations to squash actions into [0, action_max]
        transforms = torch.distributions.transforms.ComposeTransform([
            SigmoidTransform(),  # squash to (0, 1)
            AffineTransform(loc=0, scale=action_max)  # scale to (0, action_max)
        ])
        action_dist = TransformedDistribution(base_dist, transforms)
        return action_dist

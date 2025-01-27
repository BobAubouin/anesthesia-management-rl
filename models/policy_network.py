import torch
import torch.nn as nn

class HierarchicalPolicy(nn.Module):
    def __init__(self, obs_dim:int, action_dim:int):
        super().__init__()
        self.trajectory_net = nn.Sequential(
            nn.Linear(obs_dim, 64), #High-level: Generate plan for desired BIS level
            nn.ReLU() 
        )

        self.planner_net = nn.Sequential(
            nn.Linear(64,32),  #Low-level: Computes specific adjustments in the infusion rate
            nn.ReLU(),
            nn.Linear(32, action_dim)

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trajectory = self.trajectory_net(x)
        return self.planner_net(trajectory)
import pytest
import torch
import numpy as np
from models.policy_network import HierarchicalPolicy
from envs.anesthesia_env import AnesthesiaEnv

@pytest.fixture
def trained_policy():
    policy = HierarchicalPolicy(obs_dim=2, action_dim=1)
    policy.load_state_dict(torch.load("propofol_policy_final.pth"))
    return policy

@pytest.fixture
def test_env():
    return AnesthesiaEnv(config={
        'ec50': 2.7,
        'gamma': 1.4,
        'ke0': 0.46,
        'max_surgery_length': 120,
    })

#test 1: response to high BIS (should increase infusion rate)
def test_high_bis_response(trained_policy):
    torch.manual_seed(42)
    np.random.seed(42)
    state = np.array([70.0, 2.0], dtype=np.float32)
    state_tensor = torch.FloatTensor(state)

    #get action distribution
    action_dist = trained_policy(state_tensor)

    action = action_dist.sample().item()

    assert action > 5.0, (
        f"Expected relatively high infusion rate at BIS=70, got {action: .2f}"
        "Policy may not be responding to high BIS values correctly"
    )

#test 2: response to low BIS (should decrease infusion rate)
def test_low_bis_response(trained_policy):
    torch.manual_seed(42)
    np.random.seed(42)
    state = np.array([30.0, 4.0], dtype=np.float32)
    state_tensor = torch.FloatTensor(state)

    action_dist = trained_policy(state_tensor)
    action = action_dist.sample().item()

    assert action < 3.0, (
        f"Expected infusion <3.0 at BIS = 30, got {action:.2f}"
        "Policy may not be preventing overdose"
    )



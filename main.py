from envs.anesthesia_env import AnesthesiaEnv
from models.policy_network import HierarchicalPolicy

def train_agent(env, policy, episodes = 1000):
    """
    Proximal Policy Optimization (PPO) algorithm, well suited for environments
    with significant consequences can be used here.

    Args:
        env: The simulation environment.
        policy: The policy network that will learn the task.
        episodes: Number of episodes to train the agent
    """
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            policy.update(state, action, reward, next_state, done) #Implement PPO rule
            state = next_state

def main():
    config = {
        'ec50': 2.7,
        'ec50_std': 0.3,
        'gamma': 1.4,
        'ke0': 0.46,
        'obs_delay': 2,
        'action_delay': 1.0,
        'shift': 'night'
    }
    env = AnesthesiaEnv(config)
    policy = HierarchicalPolicy(obs_dim=env.observation_space.shape[0],
                                action_dim=env.action_space.shape[0])
    
    #train agent 
    metrics =train_agent(env, policy)
    print(metrics)

if __name__ == "__main__":
    main()
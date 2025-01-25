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
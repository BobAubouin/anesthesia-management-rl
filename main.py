import torch
import torch.optim as optim
import numpy as np
from envs.anesthesia_env import AnesthesiaEnv
from models.policy_network import HierarchicalPolicy

def train_agent(env: AnesthesiaEnv, policy: HierarchicalPolicy, num_episodes: int=1000) -> dict:
    """
    Train the agent using the environment and policy

    Returns:
    - a dictionary containing the training metrics
    """
    optimizer = optim.Adam(policy.parameters(), lr = 1e-3)
    gamma = 0.99
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'losses': []
    }

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length =0
        losses = []

        while not done:
            #convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            #get action from policy
            action_tensor = policy(obs_tensor) 
            action = action_tensor.detach().numpy().flatten() 

            #take step in the environment
            next_obs, reward, done, _ = env.step(action)

            #store experience ()
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0) #
            reward_tensor = torch.FloatTensor([reward]) 

            #compute TD error
            with torch.no_grad():
                next_value= policy(next_obs_tensor).max().item()
            target_value = reward_tensor + gamma * next_value
            current_value = policy(obs_tensor).max()
            td_error = target_value - current_value

            #get loss and update the policy
            loss = td_error.pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #update metrics
            episode_reward +=reward
            episode_length += 1
            losses.append(loss.item())

            #update observation
            obs = next_obs

        #log metrics for this episode
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        metrics['losses'].append(np.mean(losses))

        if (episode +1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Loss: {np.mean(losses)}")
    return metrics

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
    metrics =train_agent(env, policy, num_episodes=1000)

    #print final metrics
    print(f"Training complete. Average reward: {np.mean(metrics['episode_rewards'])}")

if __name__ == "__main__":
    main()
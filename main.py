import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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
        'losses': [],
        'bis_history': [],
        'infusion_history': []
    }
    #initialise plots
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,12)) 
    fig.suptitle('Training Progress')

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        losses = []
        bis_history = []
        infusion_history = []


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

            #store bis and infusion rate for visualisation
            bis_history.append(obs[0])
            infusion_history.append(action[0])

            #update observation
            obs = next_obs

        #log metrics for this episode
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        metrics['losses'].append(np.mean(losses))
        metrics['bis_history'].append(bis_history)
        metrics['infusion_history'].append(infusion_history)

        #plot every 10 episodes
        if (episode +1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Loss: {np.mean(losses)}")

            #clear previous plots
            ax1.clear()
            ax2.clear()
            ax3.clear()

            #plot BIS level over time
            ax1.plot(bis_history, label = 'BIS Level')
            ax1.axhline(y=40, color = 'r', linestyle = '--', label = 'BIS lower bound')
            ax1.axhline(y=60, color='r', linestyle = '--', label = 'BIS upper bound')

            ax1.set_title('BIS level over time')
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('BIS Level')
            ax1.legend()

            #plot infusion rate over time
            ax2.plot(infusion_history, label='Propofol Infusion Rate')
            ax2.set_title('Propofol Infusion Rate Over Time')
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Infusion Rate (mL/kg/min)')
            ax2.legend()

            #plot rewards over episodes
            ax3.plot(metrics['episode_rewards'], label = 'Episode Reward')
            ax3.set_title('Rewards Over Episodes')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Reward')
            ax3.legend()

            #update plots
            plt.tight_layout()
            plt.pause(0.1)
    return metrics

def main():
    config = {
        'ec50': 2.7,
        'ec50_std': 0.3,
        'gamma': 1.4,
        'ke0': 0.46,
        'obs_delay': 0, #consider removing
        'action_delay': 0.0,#consider removing
        'shift': 'night',
        'max_surgery_length': 120,
    }
    env = AnesthesiaEnv(config)
    policy = HierarchicalPolicy(obs_dim=env.observation_space.shape[0],
                                action_dim=env.action_space.shape[0])
    
    #train agent 
    metrics =train_agent(env, policy, num_episodes=1000)

    #print final metrics
    print(f"Training complete. Average reward: {np.mean(metrics['episode_rewards'])}")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
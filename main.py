import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from envs.anesthesia_env import AnesthesiaEnv
from models.policy_network import HierarchicalPolicy

def train_agent(env: AnesthesiaEnv, policy: HierarchicalPolicy, num_episodes: int=2500) -> dict:
    """
    Train the agent using REINFORCE algorithm

    Returns:
    - a dictionary containing the training metrics
    """
    optimizer = optim.Adam(policy.parameters(), lr = 1e-3)
    gamma = 0.99
    
    #training metrics storage
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'bis_history': [],
        'infusion_history': []
    }
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10,8))

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        rewards = []
        log_probs = []
        bis_trace = []
        infusion_trace = []

        while not done:
            #convert observation to tensor
            obs_tensor = torch.FloatTensor(obs)

            #get action distribution and sample
            action_dist = policy(obs_tensor)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum()

            #take action in environment
            next_obs, reward, done, _ = env.step(action.numpy())

            #store data
            rewards.append(reward)
            log_probs.append(log_prob)
            obs = next_obs
            bis_trace.append(obs[0]) #store BIS value
            infusion_trace.append(action.item()) #store infusion rate

            obs = next_obs

        #calculate discounted returns
        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8) #normalise

        #calculate policy loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R) #negative for gradient ascent
        policy_loss = torch.stack(policy_loss).sum()

        #update policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        #store metrics
        metrics['episode_rewards'].append(sum(rewards))
        metrics['episode_lengths'].append(len(rewards))
        metrics['bis_history'].append(bis_trace)
        metrics['infusion_history'].append(infusion_trace)
       

    #     #plot every 50 episodes
        if (episode +1) % 50 == 0:
            avg_bis = np.mean(bis_trace)
            avg_infusion = np.mean(infusion_trace)
            
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Avg Reward: {sum(rewards):.2f}, "
                  f"Avg BIS: {avg_bis:.2f}, "
                  f"Avg Infusion: {avg_infusion:.2f}")
            ax1.clear()
            ax2.clear()
            
            # Plot BIS values
            ax1.plot(bis_trace, label='BIS')
            ax1.axhline(40, color='r', linestyle='--', label='Safe Range')
            ax1.axhline(60, color='r', linestyle='--')
            ax1.set_title(f'Episode {episode+1} - BIS Levels')
            ax1.legend()

            # Plot infusion rates
            ax2.plot(infusion_trace, label='Infusion Rate')
            ax2.set_title('Propofol Infusion Rates')
            ax2.legend()

            plt.tight_layout()
            plt.pause(0.1)

    plt.ioff()
    plt.show()
    return metrics
    
def main():
    config = {
        'ec50': 2.7,
        'ec50_std': 0.3,
        'gamma': 1.4,
        'ke0': 0.46,
        'max_surgery_length': 120,
    }
    env = AnesthesiaEnv(config)
    policy = HierarchicalPolicy(obs_dim=env.observation_space.shape[0],
                                action_dim=env.action_space.shape[0])
    
    #train agent 
    num_episodes=2500
    metrics =train_agent(env, policy, num_episodes)

    #save final policy

    # Final metrics report
    print(f"\nTraining completed with {num_episodes} episodes")
    print(f"Average final reward (last 100 episodes): {np.mean(metrics['episode_rewards'][-100:]):.2f}")
    
    last_100_bis = np.concatenate(metrics['bis_history'][-100:])
    print(f"Average BIS (last 100 episodes): {np.nanmean(last_100_bis):.2f}")
    
    last_100_infusion = np.concatenate(metrics['infusion_history'][-100:])
    print(f"Average infusion rate (last 100 episodes): {np.nanmean(last_100_infusion):.2f} mL/kg/min")

if __name__ == "__main__":
    main()
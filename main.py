import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random

#seeding
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#hyperparameters
GAMMA = 0.9
LR = 0.01
EPISODES = 1001
MAX_STEPS = 120

#visualise
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 8))

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 10) #input: bis, effect site concentration
        # seed network initialization
        torch.manual_seed(SEED)  # Ensures consistent weight initialization
        self.layer.weight.data.normal_(0, 0.1)

    def forward(self, x):
        return torch.softmax(self.layer(x), dim = -1) #output probabilities

#initialise policy
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=LR)
possible_actions = list(range(1, 11)) #infusion is discrete numbers from 1 -10

#track training
bis_history = []
infusion_history = []

for episode in range(EPISODES):
    state = [100.0, 0.0] #starting [BIS, effect site concentration]
    episode_data = {
        'bis': [],
        'infusion': [],
        'rewards': [],
        'probabilities': []
    }
    episode_log_probs = []
    episode_rewards = []

    for step in range(MAX_STEPS):
        state_tensor = torch.FloatTensor(state)
        action_probs = policy(state_tensor) #keep gradients
        

        #action chosen randomly based on probabilities
        action_np = np.random.choice(possible_actions, p=action_probs.detach().numpy()) 

        log_prob = torch.log(action_probs[action_np-1]) #index 0-9 for actions 1-10
        episode_log_probs.append(log_prob)

        #simulate environment
        effect_site = state[1] * 0.9 + action_np #simple PK with 90% elimination as drug accumulates
        bis = max(0, min(100, 100 -2 * effect_site)) #bis decreases with increased drug with limit between 1 -100

        #calculate reward
        rewards = -abs((bis -50)* 2)
        if 40 <= bis <= 60:
            rewards += 1
        episode_rewards.append(rewards)

        #store data progression
        episode_data['bis'].append(bis)
        episode_data['infusion'].append(action_np)
        episode_data['rewards'].append(rewards)
        episode_data['probabilities'].append(action_probs)

        #update state 
        state = [bis, effect_site]

    #with calculate discounted rewards
    discounts = [GAMMA**i for i in range(len(episode_rewards))]
    discounted_rewards = [sum(g * r for g, r in zip(discounts[i:], episode_rewards[i:]))
                          for i in range(len(episode_rewards))]
    
    #convert to tensors
    discounted_rewards = torch.FloatTensor(discounted_rewards)
    log_probs = torch.stack(episode_log_probs)

    #store episode results
    bis_history.append(episode_data['bis'])
    infusion_history.append(episode_data['infusion'])

    #simple learning: increase the probability of good actions
    #convert reward to tensor 
    rewards_tensor = torch.FloatTensor(episode_data['rewards'])

    #loss = -torch.mean(rewards_tensor * log_probs)
    loss = -torch.mean(log_probs * discounted_rewards)

    #update network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #update visualisation every 50 episodes
    if episode % 50 == 0:
        ax1.clear()
        ax2.clear()

        #plot BIS 
        ax1.plot(bis_history[-1], label = 'BIS')
        ax1.axhline(40, color = 'r', linestyle = '--')
        ax1.axhline(60, color = 'r', linestyle = '--')
        ax1.set_ylabel('BIS')
        ax1.set_ylim(0, 100)
        ax1.set_title(f'Episode {episode}')

        #plot infusion rates
        ax2.plot(infusion_history[-1], label = 'Infusion Rate')
        ax2.set_ylabel('Infusion Rate mL/kg/min')
        ax2.set_xlabel('Steps (minutes)')
        ax2.set_ylim(0, 10)

        plt.tight_layout()
        plt.pause(0.1)

plt.ioff()
plt.show()
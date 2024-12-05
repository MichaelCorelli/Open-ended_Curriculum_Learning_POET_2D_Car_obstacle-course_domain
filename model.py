import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#Actor-Critic framework

#The Policy Network is responsible for determining the actions the agent should take
#based on the current state of the environment. It outputs continuous values that 
#represent the agent's decisions, such as speed and steering adjustments.
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
            
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
    
#The Value Network is used to estimate the value of a given state. The value represents 
#the expected cumulative reward the agent can achieve from that state. This is used 
#for evaluating how good a particular state is in reinforcement learning.
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

#The CarAgent encapsulates both the policy and value networks.
#   - The policy network decides actions to maximize rewards.
#   - The value network evaluates states to guide learning.
#This structure supports advanced reinforcement learning methods, such as Actor-Critic,
#where the policy and value networks work in tandem to improve agent performance.
class CarAgent:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001, weight_decay=1e-4):
        self.policy_network = PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.value_network = ValueNetwork(input_dim, hidden_dim)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr, weight_decay=weight_decay)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        #Selects an action using the policy network based on the current state.
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.policy_network(state_tensor).squeeze(0).numpy()
        return action

    def update_policy(self, rewards, states, actions, noise_std, alpha):
        #Updates the policy network using policy gradient methods to improve the agent's decisions.
        rewards = torch.FloatTensor(rewards)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)

        #Policy gradient with random perturbations
        grads = []
        for eps in torch.randn((len(actions), len(actions[0]))) * noise_std:
            perturbed_action = actions + eps
            loss = -torch.sum(rewards * self.policy_network(states + perturbed_action))
            self.policy_optimizer.zero_grad()
            loss.backward()
            grads.append([p.grad.clone() for p in self.policy_network.parameters()])

        mean_grad = torch.mean(torch.stack(grads), dim=0)
        for param, grad in zip(self.policy_network.parameters(), mean_grad):
            param.grad = grad
        self.policy_optimizer.step()

    #Updates the value network to predict the value of states.
    #So that the agent understands which states are beneficial for achieving high rewards.
    def update_value(self, states, rewards):
        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        value_pred = self.value_network(states)
        loss = self.criterion(value_pred.squeeze(), rewards)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

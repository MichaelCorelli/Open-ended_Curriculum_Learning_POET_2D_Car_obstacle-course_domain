#model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Actor-Critic framework

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

def tensor(tuple):
    return torch.stack([torch.tensor(j, dtype = torch.float32, device = device) for j in tuple])

class Buffer:

    def __init__(self, size = 20000, burn_in = 5000):
        self.size = size
        self.burn_in = burn_in
        self.buffer = namedtuple('buffer', field_names = ['state', 'action', 'reward', 'done', 'next_state'])
        self.replay = deque(maxlen = size)

    def capacity(self):
        return len(self.replay) / self.size
    
    def burn_in_capacity(self):
        return len(self.replay) / self.burn_in
    
    def append(self, s_0, a, r, d, s_1):
        self.replay.append(self.buffer(s_0, a, r, d, s_1))

    def batch(self, batch_size=32):
        indices = torch.randint(0, len(self.replay), (batch_size,))
        batch = [self.replay[i] for i in indices]
        s, a, r, d, ns = zip(*batch)
        return s, a, r, d, ns

class Q(nn.Module):
    def __init__(self, env, lr):
        super(Q, self).__init__()
        input_dim = env.observation_space.shape[0]
        hidden_dim = 128
        output_dim = env.action_space.shape[0] 
        self.lr = lr
        
        self.network = PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def greedy_a(self, state):
        q_v = self.network(state)
        action = q_v.cpu().detach().numpy()[0]
        return action

class DDQN:
    def __init__(self, env, b, lr, epsilon_initial, batch_size, threshold_r):
        self.env = env
        self.b = b
        self.lr = lr
        self.epsilon_initial = epsilon_initial
        self.batch_size = batch_size
        self.threshold_r = threshold_r
        
        self.network = Q(env, lr=self.lr).to(device)
        self.network_t = Q(env, lr=self.lr).to(device)

        self.step_c = 0
        self.rewards = 0
        self.train_rewards = []
        self.sync_eps = []
        self.train_loss = []
        self.update_loss = []
        self.train_mean_rewards = []
        self.window = 100

    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.b.batch(batch_size = self.batch_size)
        loss = self.compute_loss(batch)
        loss.backward()
        self.network.optimizer.step()
        self.update_loss.append(loss.item())

    def compute_loss(self, batch):
        s, a, rewards, done, next_s = batch
        s, a, rewards, done, next_s = map(lambda x: torch.tensor(x, device=device), [s, a, rewards, done, next_s])

        q_v = self.network(s)
        q_v = torch.gather(q_v, 1, a)
        q_v_t_1 = self.network_t(next_s)
        q_v_t_1_max = torch.max(q_v_t_1, dim = -1)[0].reshape(-1, 1)
        q_v_t = rewards + (1-done)*self.gamma*q_v_t_1_max
        loss = self.loss(q_v, q_v_t)

        return loss

    def step(self, mode = 'exploit'):
        a = self.env.action_space.sample() if mode == 'explore' else self.network.greedy_a(torch.FloatTensor(self.s_0).to(device))
        s_1, reward, terminated, truncated, _ = self.env.step(a)
        done = terminated or truncated

        self.b.append(self.s_0, a, reward, terminated, s_1)
        self.rewards += reward
        self.s_0 = s_1.copy()
        self.step_c += 1

        if done:
            self.s_0, _ = self.env.reset()

        return done

    def train(self, e_max, gamma, frequency_update, frequency_sync):
        self.e_max = e_max
        self.gamma = gamma
        self.frequency_update = frequency_update
        self.frequency_sync = frequency_sync

        self.loss = nn.MSELoss()
        self.s_0, _ = self.env.reset()

        while self.b.burn_in_capacity() < 1:
            self.step(mode = 'explore')

        e = 0
        train = True
        
        while train:
            self.s_0, _ = self.env.reset()
            self.rewards = 0
            done = False

            while not done:
                if ((e % 10) == 0):
                    self.env.render()

                p = np.random.random()
                mode = 'explore' if p < self.epsilon else 'exploit'
                done = self.step(mode)

                if self.step_c % frequency_update == 0:
                    self.update()
                if self.step_c % frequency_sync == 0:
                    self.network_t.load_state_dict(self.network.state_dict())
                    self.sync_eps.append(e)

                if done:
                    if self.epsilon >= 0.03:
                        self.epsilon *= 0.5
                    e += 1
                    reward_limit = 2000
                    self.train_rewards.append(min(self.rewards, reward_limit))
                    
                    avg_loss = np.mean(self.update_loss) if self.update_loss else 0
                    self.train_loss.append(avg_loss)
                    self.update_loss = []
                    
                    mean_rewards = np.mean(self.train_rewards[-self.window:])
                    mean_loss = np.mean(self.train_loss[-self.window:])
                    self.train_mean_rewards.append(mean_rewards)

                    if e >= e_max:
                        train = False
                        break

                    if mean_rewards >= self.reward_threshold:
                        train = False

        self.save()
        self.plot()

    def evaluate(self, eval, ep_n):
        rewards = 0

        for _ in range(ep_n):
            done = False
            s, _ = eval.reset()
            while not done:
                a = self.network.greedy_a(torch.FloatTensor(s).to(device))
                s1, r, terminated, truncated, _ = eval.step(a)
                done = terminated or truncated
                rewards += r
                s = s1

        return rewards / ep_n

    def save(self):
        torch.save(self.network, "Q")

    def load(self):
        self.network = torch.load("Q")
        self.network.eval()

    def plot(self):
        plt.plot(self.train_mean_rewards)
        plt.title('Train: mean rewards')
        plt.ylabel('reward')
        plt.xlabel('episodes')
        plt.show()
        plt.savefig('train_mean_rewards.png')
        plt.clf()
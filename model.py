import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, output_dim) 
        )

    def forward(self, x):
        return self.model(x)

class Buffer:

    def __init__(self, size=20000, burn_in=5000):
        self.size = size
        self.burn_in = burn_in
        self.buffer = namedtuple('buffer', field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay = deque(maxlen=size)

    def capacity(self):
        return len(self.replay) / self.size
    
    def burn_in_capacity(self):
        return len(self.replay) / self.burn_in
    
    def append(self, s_0, a, r, d, s_1):
        self.replay.append(self.buffer(s_0, a, r, d, s_1))

    def batch(self, batch_size=32):
        if len(self.replay) < batch_size:
            batch_size = len(self.replay)
        indices = torch.randint(0, len(self.replay), (batch_size,))
        batch = [self.replay[i] for i in indices]
        s, a, r, d, ns = zip(*batch)
        s = np.array(s)
        a = np.array(a)
        r = np.array(r)
        d = np.array(d)
        ns = np.array(ns)
        return s, a, r, d, ns

class Q(nn.Module):
    def __init__(self, env, lr):
        super(Q, self).__init__()
        input_dim = env.observation_space.shape[0]
        hidden_dim = 128
        output_dim = env.action_space.n
        self.lr = lr
        
        self.network = QNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
    
    def forward(self, x):
        return self.network(x)

    def greedy_a(self, state):
        self.network.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(dtype=torch.float32)
            q_v = self.forward(state)
            action = torch.argmax(q_v, dim=1).cpu().numpy()
        
        self.network.train()
    
        return action[0]

class DDQN:
    def __init__(self, env, b, lr, epsilon_initial, batch_size, threshold_r, render_during_training=False):
        self.env = env
        self.b = b
        self.lr = lr
        self.epsilon_initial = epsilon_initial
        self.epsilon = self.epsilon_initial
        self.batch_size = batch_size
        self.reward_threshold = threshold_r
        
        self.network = Q(env, lr=self.lr).to(device)
        self.network_t = Q(env, lr=self.lr).to(device)
        self.network_t.network.load_state_dict(self.network.network.state_dict())
        self.network_t.network.eval()

        self.step_c = 0
        self.rewards = 0
        self.train_rewards = []
        self.sync_eps = []
        self.train_loss = []
        self.update_loss = []
        self.train_mean_rewards = []
        self.window = 100
        self.render_during_training = render_during_training

        self.gamma = 0.99

    def update(self):
        self.network.network.train()
        self.network.optimizer.zero_grad()
        batch = self.b.batch(batch_size=self.batch_size)
        loss = self.compute_loss(batch)
        loss.backward()
        self.network.optimizer.step()
        self.update_loss.append(loss.item())

    def compute_loss(self, batch):
        s, a, rewards, done, next_s = batch
        s = torch.from_numpy(s).to(device, dtype=torch.float32)
        a = torch.from_numpy(a).to(device, dtype=torch.long).unsqueeze(-1) 
        rewards = torch.from_numpy(rewards).to(device, dtype=torch.float32).unsqueeze(-1)
        done = torch.from_numpy(done).to(device, dtype=torch.float32).unsqueeze(-1)
        next_s = torch.from_numpy(next_s).to(device, dtype=torch.float32)

        q_v = self.network.network(s)
        q_v = torch.gather(q_v, 1, a)
        q_v_t_1 = self.network_t.network(next_s)
        q_v_t_1_max = torch.max(q_v_t_1, dim=1, keepdim=True)[0]
        q_v_t = rewards + (1 - done) * self.gamma * q_v_t_1_max
        loss = self.loss(q_v, q_v_t)
        return loss

    def step(self, mode='exploit'):
        if mode == 'explore':
            a = self.env.action_space.sample()
        else:
            a = self.network.greedy_a(torch.FloatTensor(self.s_0).to(device))
        s_1, reward, done, info = self.env.step(a)

        self.b.append(self.s_0, a, reward, done, s_1)
        self.rewards += reward
        self.s_0 = s_1.copy()
        self.step_c += 1

        if done:
            self.s_0, _ = self.env.reset()

        return done

    def train(self, e_max, gamma, frequency_update, frequency_sync):
        self.gamma = gamma
        self.e_max = e_max
        self.frequency_update = frequency_update
        self.frequency_sync = frequency_sync

        self.loss = nn.MSELoss()
        self.s_0, _ = self.env.reset()

        while self.b.burn_in_capacity() < 1:
            self.step(mode='explore')

        e = 0
        train = True
        
        with tqdm(total=self.e_max, desc="Training Progress") as pbar:
            while train:
                self.s_0, _ = self.env.reset()
                self.rewards = 0
                done = False

                while not done:
                    if ((e % 10) == 0) and self.render_during_training:
                        self.env.render()

                    p = np.random.random()
                    mode = 'explore' if p < self.epsilon else 'exploit'
                    done = self.step(mode)

                    if self.step_c % frequency_update == 0:
                        self.update()
                    if self.step_c % frequency_sync == 0:
                        self.network_t.network.load_state_dict(self.network.network.state_dict())
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
                        pbar.update(1)

                        if e >= e_max:
                            train = False
                            break

                        if mean_rewards >= self.reward_threshold:
                            train = False

        self.save()
        self.plot()

    def evaluate(self, eval_env, ep_n=1, render=False, verbose=True, print_results=False):
        self.network.network.eval()
        episode_rewards = []
        episode_steps = []
        final_positions = []
        max_steps_eval = 100
        
        with torch.no_grad():
            for e_i in range(ep_n):
                s, info = eval_env.reset()
                done = False
                ep_reward = 0
                steps = 0

                while not done and steps < max_steps_eval:
                    if render:
                        eval_env.render()
                    a = self.network.greedy_a(torch.FloatTensor(s).unsqueeze(0).to(device))
                    s, r, done, info = eval_env.step(a)
                    ep_reward += r
                    steps += 1

                episode_rewards.append(ep_reward)
                episode_steps.append(steps)
                
                if hasattr(eval_env, 'car') and hasattr(eval_env.car, 'body'):
                    final_positions.append(eval_env.car.body.position[0])  

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)

        mean_steps = np.mean(episode_steps)
        std_steps = np.std(episode_steps)

        metrics = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "mean_steps": mean_steps,
            "std_steps": std_steps
        }

        if final_positions:
            metrics["mean_final_position"] = np.mean(final_positions)
            metrics["min_final_position"] = np.min(final_positions)
            metrics["max_final_position"] = np.max(final_positions)

        if print_results:
            print("----- Evaluation Results -----")
            print(f"Episodes: {ep_n}")
            print(f"Mean Reward: {mean_reward:.2f}")
            print(f"Std Reward: {std_reward:.2f}")
            print(f"Min/Max Reward: {min_reward:.2f}/{max_reward:.2f}")
            if final_positions:
                print(f"Final Position (X) - Mean: {metrics['mean_final_position']:.2f}, "
                    f"Min: {metrics['min_final_position']:.2f}, "
                    f"Max: {metrics['max_final_position']:.2f}")
            print("--------------------------------")

        return metrics


    def save(self):
        
        torch.save({
            'network_state_dict': self.network.network.state_dict(),
            'target_network_state_dict': self.network_t.network.state_dict(),
            'optimizer_state_dict': self.network.optimizer.state_dict(),
        }, "DDQN_state_dict.pth")

    def load(self):
        checkpoint = torch.load("DDQN_state_dict.pth")
        self.network.network.load_state_dict(checkpoint['network_state_dict'])
        self.network_t.network.load_state_dict(checkpoint['target_network_state_dict'])
        self.network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.network.network.eval()
        self.network_t.network.eval()

    def plot(self):
        plt.plot(self.train_mean_rewards)
        plt.title('Train: Mean Rewards')
        plt.ylabel('Reward')
        plt.xlabel('Episodes')
        plt.savefig('train_mean_rewards.png')
        plt.close()

    def plot_losses(self):
        plt.plot(self.train_loss)
        plt.title('Train: Loss')
        plt.ylabel('Loss')
        plt.xlabel('Updates')
        plt.savefig('train_loss.png')
        plt.close()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from DQNAgent.QNetwork import QNetwork
from DQNAgent.ReplayBuffer import ReplayBuffer


class DQNAgent:
    def __init__(self, action_size=3):

        # Detect MPS (Apple Silicon GPU), fall back to CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.action_size = action_size

        # Hyperparameters
        self.alpha = 0.0001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9993

        self.batch_size = 128
        self.min_buffer_size = 5000
        self.tau = 0.005          # soft update rate (Polyak averaging)
        self.steps_done = 0

        # CNN networks
        self.online_net = QNetwork(grid_size=20, output_size=action_size).to(self.device)
        self.target_net = QNetwork(grid_size=20, output_size=action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        #Optimizer, loss function, replay buffer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.alpha)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(capacity=100_000)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        self.online_net.train()

        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.min_buffer_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        predicted_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online net selects the action, target net evaluates it.
            # Prevents overestimation bias from using the same network for both.
            next_actions = self.online_net(next_states).argmax(1)
            max_next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(predicted_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        #Added this as it was observed that the loss was exploding and the agent was not learning. This is a common technique to stabilize training.
        #During loss.backward(), PyTorch computes how much each weight contributed to the error and assigns it a gradient — a number saying "change this weight by this much." The optimizer then uses those gradients to nudge weights in the right direction.
        #What clip_grad_norm_ Does
        #It's a safety ceiling. Before the optimizer takes its step, it measures the total size of all gradients combined (the "norm") and if it exceeds max_norm=10, it scales all gradients down proportionally so the total equals exactly 10:
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10)  # ← add this
        self.optimizer.step()

        self.steps_done += 1
        # Soft (Polyak) target update every step — smoother than hard copy every N steps
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1 - self.tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
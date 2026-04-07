import numpy as np
import torch
import torch.nn as nn

from DQNAgent.QNetwork import QNetwork
from DQNAgent.ReplayBuffer import ReplayBuffer
from utils.constants import GRID_SIZE


class DQNAgent:
    def __init__(
        self,
        grid_size=GRID_SIZE,
        n_actions=3,
        device=None,
        lr=1e-4,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=200_000,
        batch_size=64,
        buffer_capacity=100_000,
        learning_starts=10_000,
        target_update=1_000,
        train_every=4,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update = target_update
        self.train_every = train_every

        # ── Networks ──────────────────────────────────────────────────────────
        self.online_net = QNetwork(grid_size, n_actions).to(self.device)
        self.target_net = QNetwork(grid_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # never updated by backprop — only by hard copy

        # ── Optimizer & Loss ──────────────────────────────────────────────────
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        # Huber loss: behaves like MSE near 0, like MAE for large errors
        # More robust than pure MSE when Q-value estimates are noisy early on
        self.loss_fn = nn.SmoothL1Loss()

        # ── Replay Buffer ─────────────────────────────────────────────────────
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        # ── Epsilon Tracking ──────────────────────────────────────────────────
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.epsilon = eps_start

        # ── Counters ──────────────────────────────────────────────────────────
        self.total_steps = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Epsilon Decay (linear)
    # ──────────────────────────────────────────────────────────────────────────

    def _update_epsilon(self):
        progress = min(self.total_steps / self.eps_decay_steps, 1.0)
        self.epsilon = self.eps_start + progress * (self.eps_end - self.eps_start)

    # ──────────────────────────────────────────────────────────────────────────
    # Action Selection
    # ──────────────────────────────────────────────────────────────────────────

    def choose_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy: explore randomly or exploit learned Q-values."""
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))

        with torch.no_grad():
            # Normalize from {0,1,2,3} → [0,1] before passing to CNN
            state_t = torch.FloatTensor(state / 3.0).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)   # shape: (1, 3)
            return int(q_values.argmax(dim=1).item())

    # ──────────────────────────────────────────────────────────────────────────
    # Learning Step (Bellman Update)
    # ──────────────────────────────────────────────────────────────────────────

    def learn(self) -> float | None:
        """One gradient update using a random batch from the replay buffer."""
        if len(self.buffer) < self.learning_starts:
            return None  # wait until buffer has enough diverse experience

        states, actions, rewards, next_states, terminateds = self.buffer.sample(self.batch_size)

        # Move everything to GPU/CPU and normalize pixel-like grid values
        states      = (states      / 3.0).to(self.device)
        next_states = (next_states / 3.0).to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        terminateds = terminateds.to(self.device)

        # ── Current Q(s, a) — only for the action actually taken ──────────────
        q_all_actions = self.online_net(states)                   # (B, 3)
        q_current = q_all_actions.gather(
            1, actions.unsqueeze(1)                               # select column per row
        ).squeeze(1)                                              # (B,)

        # ── Target: r + γ * max_a' Q_target(s', a')  [0 if terminal] ─────────
        with torch.no_grad():
            q_next_max = self.target_net(next_states).max(dim=1).values  # (B,)
            q_target = rewards + self.gamma * q_next_max * (1.0 - terminateds)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients: prevents instability when Q-targets jump early in training
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    # Backward-compatible alias for older training scripts.
    def train(self) -> float | None:
        return self.learn()

    # ──────────────────────────────────────────────────────────────────────────
    # Target Network Sync
    # ──────────────────────────────────────────────────────────────────────────

    def update_target(self):
        """Hard copy: online weights → target weights."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ──────────────────────────────────────────────────────────────────────────
    # Master Step (call this once per env step in your training loop)
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, state, action, reward, next_state, terminated) -> float | None:
        """
        Orchestrates everything for one env step:
          store → (maybe) learn → (maybe) sync target → decay epsilon
        Returns the loss value if a learn step happened, else None.
        """
        self.buffer.store(state, action, reward, next_state, terminated)
        self.total_steps += 1
        self._update_epsilon()

        loss = None
        if self.total_steps % self.train_every == 0:
            loss = self.learn()

        if self.total_steps % self.target_update == 0:
            self.update_target()

        return loss

    # ──────────────────────────────────────────────────────────────────────────
    # Checkpoint Save / Load
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path="checkpoints/dqn_snake.pth"):
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "epsilon":     self.epsilon,
        }, path)
        print(f"[Saved] {path}  (step={self.total_steps}, ε={self.epsilon:.3f})")

    def load(self, path="checkpoints/dqn_snake.pth"):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]
        self.epsilon     = ckpt["epsilon"]
        print(f"[Loaded] {path}  (step={self.total_steps}, ε={self.epsilon:.3f})")


# ─────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────

if __name__ == "__main__":
    agent = DQNAgent()
    print(f"Device: {agent.device}")
    print(f"Online net:\n{agent.online_net}")

    dummy_state = np.random.randint(0, 4, size=(400,)).astype(np.float32)
    action = agent.choose_action(dummy_state)
    print(f"\nSample action (ε=1.0, fully random): {action}")

    # Verify a learn step works end-to-end
    for _ in range(10_001):  # fill past learning_starts threshold
        s  = np.random.rand(400).astype(np.float32)
        a  = np.random.randint(3)
        r  = float(np.random.choice([-10, -0.01, 10]))
        s2 = np.random.rand(400).astype(np.float32)
        d  = bool(np.random.rand() > 0.95)
        agent.buffer.store(s, a, r, s2, d)

    loss = agent.learn()
    print(f"Loss after first learn step: {loss:.6f}")
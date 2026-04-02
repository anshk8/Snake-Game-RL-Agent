
'''
    What is Replay buffer?
- A replay buffer is a data structure that stores the agent's experiences (state, action, reward, next_state, done) during training.
- It allows the agent to learn from past experiences by sampling random batches of these experiences during the training process.


'''


from collections import deque
import random
import torch
import numpy as np


class ReplayBuffer:

    def __init__(self, capacity=100_000):
        #Max size: once full, oldest experiences are overwritten (deque handles this).
        self.buffer = deque(maxlen=capacity)


    def store(self, state, action, reward, next_state, terminated):
        #Store a new experience tuple in the buffer.
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, terminateds = zip(*batch)

        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,               dtype=torch.long),
            torch.tensor(rewards,               dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(terminateds,                 dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

if __name__ == "__main__":
    buf = ReplayBuffer(capacity=1000)

    # Store 200 fake experiences
    for _ in range(200):
        s  = np.random.rand(400).astype(np.float32)
        a  = np.random.randint(3)
        r  = float(np.random.choice([-10, 0, 10]))
        s2 = np.random.rand(400).astype(np.float32)
        d  = bool(np.random.rand() > 0.9)
        buf.store(s, a, r, s2, d)

    print(f"Buffer size: {len(buf)}")

    states, actions, rewards, next_states, dones = buf.sample(64)
    print(f"states shape:      {states.shape}")       # (64, 400)
    print(f"actions shape:     {actions.shape}")      # (64,)
    print(f"rewards shape:     {rewards.shape}")      # (64,)
    print(f"next_states shape: {next_states.shape}")  # (64, 400)
    print(f"dones shape:       {dones.shape}")        # (64,)
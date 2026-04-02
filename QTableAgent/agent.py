import numpy as np


'''
Notes/Learnings and Approach for this agent.
Notes/Learnings and Approach for this agent.
- First thing to note is that the space is not continuous but discrete and small enough for a Q-table. 


Q-Table: Is a spreadsheet of knowledge
    - Rows are states
    - Columns are actions, 
    - Each cell answers: "if I'm in this state and take this action, how much total future reward can I expect?"
    - The Q-value Q(s,a) represents the expected total discounted future reward if you take action a in state s.

    
Epsilon (ε) — Exploration vs Exploitation
    - With probability ε, choose a random action (explore).
    - With probability 1-ε, choose the action with the highest Q-value for the current state (exploit).
    - ε starts high (e.g., 1.0) and decays over time to encourage more exploitation as the agent learns and more exploration early on.

    
Alpha (α) — Learning Rate
    - Controls how much each new experience overwrites old knowledge
    EX: 
        α = 1.0 → completely replace old estimate with new one (forgets everything)
        α = 0.0 → never update (learns nothing)

        
Gamma (γ) — Discount Factor
    - Controls how much the agent cares about future rewards vs immediate rewards:
    EX:
        γ = 0.0 → only cares about immediate reward
        γ close to 1.0 → values future rewards almost as much as immediate rewards

        

WHY WE CAN'T WIN / ARE LIMITED:
    - State space is small enough for a Q-table (4096 states × 3 actions = 12,288 entries).
    - Max Score was around 30-40 range (see videos folder to watch). The agent does not know where its tail is, our state has 12 items and if we add where tail is, we will have TOO many states which is not feasible for a Q-table. 
    - So a DQN would be appropriate for a more complex state representation, see DQNAgent folder

'''


class QTableAgent:

    def __init__(self, state_size=12, action_size=3):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.alpha = 0.1        # learning rate
        self.gamma = 0.9        # discount factor
        self.epsilon = 1.0      # start fully random
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Q-table: 4096 states × 3 actions, all zeros
        self.q_table = np.zeros((2 ** state_size, action_size))

    def state_to_index(self, state):
        # Convert 12-value binary array → integer index
        # e.g. [1,0,1,...] → 101... in binary → integer
        return int("".join(str(int(x)) for x in state), 2)


    def choose_action(self, state):
        state_idx = self.state_to_index(state)

        if np.random.rand() < self.epsilon:
            # Explore — pick random action
            return np.random.randint(self.action_size)
        else:
            # Exploit — pick best known action
            return np.argmax(self.q_table[state_idx])

    def update(self, state, action, reward, next_state, terminated):
        state_idx      = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)

        # What we currently think this action is worth
        current_q = self.q_table[state_idx, action]

        # What we should update toward (Bellman target)
        if terminated:
            # No future — agent is dead, only immediate reward matters
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])

        # TD error — gap between estimate and target
        td_error = target - current_q

        # Nudge Q-value toward target by alpha
        self.q_table[state_idx, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# add to bottom of agent.py
if __name__ == "__main__":
    agent = QTableAgent()
    dummy_state      = np.array([1,0,0,0, 0,1,0,0, 1,0,0,0], dtype=np.float32)
    dummy_next_state = np.array([0,0,0,0, 0,1,0,0, 0,1,0,0], dtype=np.float32)

    print("Before update:", agent.q_table[agent.state_to_index(dummy_state)])
    agent.update(dummy_state, action=1, reward=10.0, next_state=dummy_next_state, terminated=False)
    print("After update: ", agent.q_table[agent.state_to_index(dummy_state)])
    print("Epsilon:", agent.epsilon)
    agent.decay_epsilon()
    print("Epsilon after decay:", agent.epsilon)
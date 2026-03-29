import gymnasium as gym
import numpy as np


'''
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

'''


class QAgent:

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




if __name__ == "__main__":
    agent = QAgent()
    print("Q-table shape:", agent.q_table.shape)
    print("All zeros state:", agent.state_to_index([0]*12))
    print("All ones state:", agent.state_to_index([1]*12))
    print("Mixed state:", agent.state_to_index([1,0,1,0,0,0,1,1,0,0,0,1]))

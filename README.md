# Snake-Game-RL-Agent

## Board
- 20 X 20 Grid
- One Food Spawn at a time


## Methods Used / Learned

- **Q-Table**: Inside `QTableAgent`
  - Achieved best score between 30-40
  - Limited to progress more because the enviornment does not provide the tail/body location within the state. Having this in the state would make it too big for a Q-Table
 
- **DQN Network**: Contains a ReplayBuffer, CNN with Pytorch used in training, all inside `DQNAgent`
  - Achieved Highest Score of 2 so far training on 5000 episodes
  
# TODO: (As of Apr 7, 2026)
- Look more deeply into DQNAgent, try to get much better performance.
- Use Optuna to choose Hyperparemeters (Gamma, Epsilon, Learning Rate etc)
- Change QNetwork layers IF NEEDED (likely not tbh)
- Maybe Double DQN Network 


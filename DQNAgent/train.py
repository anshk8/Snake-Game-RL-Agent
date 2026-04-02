


'''
    NOTES and LEARNINGS?


Why am I using a DQN instead of a Q-table?
- Q-Table 12-feature state vector is the bottleneck since it only shows danger in 4 dirs, food direction and current direction
- Q-Table doesn't have awareness of tail length or shape so as SNAKE grows, it occupies lots of the board
- To progress higher, we need the agent to have full grid representation  BUT catch is Q-table no longer works as WAYY too many states exist (4^400)


What will DQN do?
- DQN will use a neural network to approximate the Q-function, allowing us to handle a much larger state space (the full grid representation) without needing to store a massive Q-table.
- A convolutional neural network processes the 20×20 grid spatially — it learns to recognize patterns like "tail blocking path" or "food in corner" the same way humans see them visually



'''
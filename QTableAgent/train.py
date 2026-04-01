import numpy as np
import pickle
import json
import os
from .agent import QTableAgent
from game.snake_env import SnakeEnv

EPISODES = 2000

def train():
    env   = SnakeEnv(render_mode="none")
    agent = QTableAgent()

    scores   = []
    epsilons = []
    top_3_games = []  # Track top 3: [(episode, score, seed), ...]

    for episode in range(1, EPISODES + 1):
        state, info = env.reset(seed=episode)  # Use episode number as seed
        done = False
        total_reward = 0
        seed = info.get("seed", episode)

        while not done:
            action                              = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done                                = terminated or truncated

            agent.update(state, action, reward, next_state, terminated)
            state       = next_state
            total_reward += reward

        agent.decay_epsilon()
        score = info["score"]
        scores.append(score)
        epsilons.append(agent.epsilon)

        # Track top 3 scores
        top_3_games.append((episode, score, seed))
        top_3_games = sorted(top_3_games, key=lambda x: x[1], reverse=True)[:3]

        # Log every 100 episodes
        if episode % 100 == 0:
            avg = np.mean(scores[-100:])
            print(f"Episode {episode:4d} | Avg Score (last 100): {avg:.2f} | ε: {agent.epsilon:.3f}")

    env.close()
    
    # Save the trained agent
    os.makedirs("models", exist_ok=True)
    with open("models/trained_agent.pkl", "wb") as f:
        pickle.dump(agent, f)
    print(f"\nAgent saved to models/trained_agent.pkl")
    
    # Save top 3 games info as JSON
    top_3_data = [
        {"rank": i+1, "episode": ep, "score": score, "seed": seed}
        for i, (ep, score, seed) in enumerate(top_3_games)
    ]
    with open("models/top_3_games.json", "w") as f:
        json.dump(top_3_data, f, indent=2)
    print(f"Top 3 games info saved to models/top_3_games.json")
    
    return scores, epsilons, top_3_games

if __name__ == "__main__":
    scores, epsilons, top_3_games = train()
    print(f"\nTraining complete.")
    print(f"First 100 episodes avg:  {np.mean(scores[:100]):.2f}")
    print(f"Last  100 episodes avg:  {np.mean(scores[-100:]):.2f}")
    print(f"\nTop 3 highest scores:")
    for rank, (ep, score, seed) in enumerate(top_3_games, 1):
        print(f"  {rank}. Episode {ep}: Score {score} (seed: {seed})")
# train_dqn.py
# Run with: python -m DQNAgent.trainDQNAgent

from pathlib import Path
import json

import numpy as np
import torch

from game.snake_env_DQN import SnakeEnv
from DQNAgent.DQNAgent import DQNAgent


'''
    NOTES and LEARNINGS?


Why am I using a DQN (Deep Q-Network ) instead of a Q-table?
- Q-Table 12-feature state vector is the bottleneck since it only shows danger in 4 dirs, food direction and current direction
- Q-Table doesn't have awareness of tail length or shape so as SNAKE grows, it occupies lots of the board
- To progress higher, we need the agent to have full grid representation  BUT catch is Q-table no longer works as WAYY too many states exist (4^400)


What will DQN (Deep Q-Network ) do?
- DQN will use a neural network to approximate the Q-function, allowing us to handle a much larger state space (the full grid representation) without needing to store a massive Q-table.
- A convolutional neural network processes the 20×20 grid, it learns to recognize patterns like "tail blocking path" or "food in corner" the same way humans see them visually

ReplayBuffer    — stores experiences, samples random batches
QNetwork        — the neural net: Input→Hidden→Hidden→Output(3)
DQNAgent        — ties it together: uses QNetwork + ReplayBuffer, choose_action() and update() like before but with gradients


'''


# --- Config ---
NUM_EPISODES = 5_000
MAX_STEPS = 2_000     # safety cap per episode (prevent infinite loops)
PRINT_EVERY = 50      # print stats every N episodes

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"
MODEL_PATH = MODEL_DIR / "dqn_agent.pth"
TOP_GAMES_PATH = MODEL_DIR / "top_3_games.json"


def _save_model(agent):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(agent.online_net.state_dict(), MODEL_PATH)


def _save_top_games(top_3_games):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "rank": index + 1,
            "episode": game["episode"],
            "score": game["score"],
            "seed": game["seed"],
        }
        for index, game in enumerate(top_3_games)
    ]
    with TOP_GAMES_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def train(num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, print_every=PRINT_EVERY):
    env = SnakeEnv(render_mode="none")
    agent = DQNAgent()

    episode_scores = []
    episode_rewards = []
    top_3_games = []

    for episode in range(1, num_episodes + 1):

        obs, info = env.reset(seed=episode)
        seed = info.get("seed", episode)
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            agent.step(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += reward

            if done:
                break

        episode_scores.append(info["score"])
        episode_rewards.append(total_reward)

        top_3_games.append(
            {
                "episode": episode,
                "score": info["score"],
                "seed": seed,
            }
        )
        top_3_games = sorted(top_3_games, key=lambda game: game["score"], reverse=True)[:3]

        # --- Logging ---
        if episode % print_every == 0:
            avg_score = np.mean(episode_scores[-print_every:])
            avg_reward = np.mean(episode_rewards[-print_every:])
            print(
                f"Episode {episode:>5} | "
                f"Avg Score: {avg_score:>5.2f} | "
                f"Avg Reward: {avg_reward:>7.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.buffer):>6}"
            )

    env.close()

    _save_model(agent)
    _save_top_games(top_3_games)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved top 3 game metadata to {TOP_GAMES_PATH}")
    return episode_scores, episode_rewards


if __name__ == "__main__":
    scores, rewards = train()
    print("\nTraining complete!")
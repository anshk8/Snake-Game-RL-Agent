from pathlib import Path
import json

import cv2
import numpy as np
import pygame
import torch

from game.snake_env_DQN import SnakeEnv
from DQNAgent.DQNAgent import DQNAgent
from utils.constants import WIDTH, HEIGHT

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
VIDEO_DIR = BASE_DIR / "videos"
MODEL_PATH = MODEL_DIR / "dqn_agent.pth"
TOP_GAMES_PATH = MODEL_DIR / "top_3_games.json"


def load_agent(model_path=MODEL_PATH):
    agent = DQNAgent()
    state_dict = torch.load(model_path, map_location="cpu")
    agent.online_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(state_dict)
    agent.online_net.eval()
    agent.target_net.eval()
    agent.epsilon = 0.0
    return agent


def load_top_3_games(info_path=TOP_GAMES_PATH):
    with info_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def pygame_surface_to_cv2(pygame_surface):
    frame = pygame.surfarray.array3d(pygame_surface)
    frame = np.transpose(frame, (1, 0, 2))
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def replay_and_record(agent, game_info, output_path, max_steps=2_000):
    pygame.init()
    env = SnakeEnv(render_mode="human")
    state, _ = env.reset(seed=game_info["seed"])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (WIDTH, HEIGHT))

    done = False
    steps = 0

    print(
        f"Recording Top 1 (Episode {game_info['episode']}, Score: {game_info['score']})...",
        end="",
        flush=True,
    )

    while not done and steps < max_steps:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env.render()
        if env.screen is not None:
            frame = pygame_surface_to_cv2(env.screen)
            video_writer.write(frame)

        state = next_state
        steps += 1

    video_writer.release()
    env.close()

    print(" Done!")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")
    if not TOP_GAMES_PATH.exists():
        raise FileNotFoundError(f"Missing top games metadata: {TOP_GAMES_PATH}")

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    agent = load_agent()
    top_games = load_top_3_games()

    if not top_games:
        print("No top games found.")
        return

    # Record only the top game
    game_info = top_games[0]
    output_path = VIDEO_DIR / f"top_gameplay_episode_{game_info['episode']}_score_{game_info['score']}.mp4"
    replay_and_record(agent, game_info, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

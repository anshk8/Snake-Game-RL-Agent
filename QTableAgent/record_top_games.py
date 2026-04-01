"""
Replay and record the top 3 highest-scoring games as videos.
This script uses the exact seeds from training to replay the same games.
Run this after training: python record_top_games.py
"""

import numpy as np
import pickle
import json
import os
import cv2
import pygame
from .agent import QTableAgent
from game.snake_env import SnakeEnv
from utils.constants import WIDTH, HEIGHT


def load_agent(model_path="models/trained_agent.pkl"):
    """Load the trained agent from disk."""
    with open(model_path, "rb") as f:
        agent = pickle.load(f)
    # Set epsilon to 0 for pure exploitation (no exploration)
    agent.epsilon = 0.0
    return agent


def load_top_3_games(info_path="models/top_3_games.json"):
    """Load the top 3 games info from training."""
    with open(info_path, "r") as f:
        return json.load(f)


def pygame_surface_to_cv2(pygame_surface):
    """Convert pygame surface to OpenCV-compatible numpy array (BGR format)."""
    # Pygame surfaces are in RGB format, pygame.surfarray returns in (width, height, 3)
    frame = pygame.surfarray.array3d(pygame_surface)
    # Rotate to (height, width, 3) and convert RGB to BGR for OpenCV
    frame = np.transpose(frame, (1, 0, 2))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def replay_and_record(agent, game_info, output_path):
    """
    Replay a specific game using its seed and record it as MP4.
    
    Args:
        agent: Trained QTableAgent
        game_info: Dict with "rank", "episode", "score", "seed"
        output_path: Path to save the video file
    """
    env = SnakeEnv(render_mode="human")
    state, info = env.reset(seed=game_info["seed"])
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10  # matches the render tick rate in env
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH, HEIGHT))
    
    frame_count = 0
    actual_score = 0
    done = False
    
    rank = game_info["rank"]
    episode = game_info["episode"]
    expected_score = game_info["score"]
    
    print(f"  Recording Top {rank} (Episode {episode}, Expected Score: {expected_score})...", 
          end="", flush=True)
    
    while not done:
        # Agent chooses action (greedy, no exploration)
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render and capture frame
        env.render()
        frame = pygame_surface_to_cv2(env.screen)
        video_writer.write(frame)
        frame_count += 1
        
        state = next_state
        actual_score = info["score"]
    
    video_writer.release()
    env.close()
    
    print(f" Done! ({frame_count} frames, actual score: {actual_score})")
    return actual_score


def main():
    # Check if trained agent exists
    if not os.path.exists("models/trained_agent.pkl"):
        print("Error: Trained agent not found at models/trained_agent.pkl")
        print("Please run 'python train.py' first to train the agent.")
        return
    
    if not os.path.exists("models/top_3_games.json"):
        print("Error: Top 3 games info not found at models/top_3_games.json")
        print("Please run 'python train.py' first to generate this file.")
        return
    
    print("Loading trained agent...")
    agent = load_agent()
    
    print("Loading top 3 games info from training...")
    top_3_games = load_top_3_games()
    
    # Create output directory
    os.makedirs("videos", exist_ok=True)
    
    print("\nReplaying and recording the exact top 3 games from training:\n")
    
    for game_info in top_3_games:
        rank = game_info["rank"]
        output_file = f"videos/top_{rank}_game.mp4"
        actual_score = replay_and_record(agent, game_info, output_file)
        print(f"       Saved to: {output_file}")
    
    print("\n✓ All videos saved to 'videos/' directory")
    print("\nYou can now watch the agent play its best games!")


if __name__ == "__main__":
    main()


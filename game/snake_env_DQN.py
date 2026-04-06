# snake_end_DQN.py: Is the env for the DQN agent, it is the same as snake_env.py but with a few changes to accomodate the DQN agent.
#Run with: python -m game.snake_env_DQN

import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import pygame
from game.player import Snake
from game.food import Food
from utils.constants import WIDTH, HEIGHT, GRID_SIZE, CELL_SIZE

# Clockwise order — used to compute relative turns
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT


#gym.Env is the base class for all Gym environments, will let us use env with RL libraries
class SnakeEnv(gym.Env):
    """
    Custom Gymnasium environment for Snake.
    The agent observes a 12-value binary state vector,
    chooses 1 of 3 relative actions, and receives rewards
    for eating food (+10) or dying (-10).
    """

    #"human" means visual window, "none" means headless (for fast training).
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, render_mode="none"):
        super().__init__()

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        #Action space has 3 actions: 0=straight, 1=turn right, 2=turn left
        self.action_space = spaces.Discrete(3)

        # --- Observation space ---
        self.observation_space = spaces.Box(
            low=0, high=3,
            shape=(GRID_SIZE * GRID_SIZE,),  # 400 flat values
            dtype=np.float32
        )

        # --- Game state (set properly in reset()) ---
        self.snake = None
        self.food = None
        self.steps_since_food = 0



    def get_state(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Mark body segments
        for (x, y) in self.snake.body[1:]:
            grid[y][x] = 1.0

        # Mark head
        hx, hy = self.snake.body[0]
        grid[hy][hx] = 2.0

        # Mark food
        fx, fy = self.food.position
        grid[fy][fx] = 3.0

        return grid.flatten()

    def _terminal_state(self):
        return np.zeros((GRID_SIZE * GRID_SIZE,), dtype=np.float32)


    #To reset the environment to an initial state 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # handles seeding for reproducibility
        
        # Seed Python's random module for deterministic food placement
        if seed is not None:
            random.seed(seed)

        # Fresh game objects every episode
        self.snake = Snake()
        self.food = Food(self.snake.body)
        self.steps_since_food = 0

        # Return initial state and info dict with seed
        observation = self.get_state()
        return observation, {"seed": seed}


    def step(self, action):
        # --- your existing direction + move logic (unchanged) ---
        current_index = DIRECTIONS.index(self.snake.direction)
        if action == 0:
            new_index = current_index
        elif action == 1:
            new_index = (current_index + 1) % 4
        else:
            new_index = (current_index - 1) % 4

        self.snake.direction = DIRECTIONS[new_index]
        self.snake.move()

        # --- reward logic ---
        reward = 0.0
        terminated = False

        if self.snake.is_dead():
            reward = -10.0
            terminated = True
        elif self.snake.body[0] == self.food.position:
            reward = 10.0
            self.snake.grow()
            self.food = Food(self.snake.body)
            self.steps_since_food = 0
        else:
            self.steps_since_food += 1
            # small time penalty discourages circling
            reward = -0.01  

            # Kill episode if snake hasn't eaten in too long (prevents infinite loops)
            if self.steps_since_food > (GRID_SIZE * GRID_SIZE) * 2:
                reward = -10.0
                terminated = True

        # --- rest of step unchanged ---
        observation = self._terminal_state() if terminated else self.get_state()
        info = {
            "score": len(self.snake.body) - 3,
            "won": len(self.snake.body) == GRID_SIZE * GRID_SIZE
        }
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "none":
            return

        # Initialize pygame on first render call
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Snake - RL Agent")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24)

        # Handle window close button
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        # Draw everything
        self.screen.fill((0, 0, 0))

        # Grid lines
        GREY = (40, 40, 40)
        for x in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GREY, (0, y), (WIDTH, y))

        # Food
        fx, fy = self.food.position
        pygame.draw.rect(self.screen, (200, 0, 0),
                        (fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Snake
        for i, (x, y) in enumerate(self.snake.body):
            color = (0, 220, 0) if i == 0 else (0, 160, 0)  # head brighter
            pygame.draw.rect(self.screen, color,
                            (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Score
        score = len(self.snake.body) - 3
        text = self.font.render(f"Score: {score}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

        pygame.display.update()
        self.clock.tick(10)  # control render speed


    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

if __name__ == "__main__":
    env = SnakeEnv(render_mode="human")
    obs, _ = env.reset()
    print('New state shape:', obs.shape)
    print('Sample values:', obs[:20])
    print('Unique values:', set(obs.tolist()))
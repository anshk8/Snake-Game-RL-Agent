'''
    A env for RL needs 3 main methods:

    1. reset() - resets the environment to an initial state and returns that state
    2. step(action) - takes an action and updates the environment, returning (new_state, reward, done)
    3. render() - draws the current state of the environment

    
    To model state will go with this, 12 different 

    state = [
    # Danger in each direction (1 = wall or body, 0 = safe)
    danger_up, danger_down, danger_left, danger_right,

    # Food direction relative to head (1 = food is in that direction)
    food_up, food_down, food_left, food_right,

    # Current movement direction (one-hot encoded)
    dir_up, dir_down, dir_left, dir_right
]

'''

import numpy as np
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
        # 12 binary values, each either 0.0 or 1.0
        #A box is "a multi-dimensional array where each value has a min and max."
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(12,), dtype=np.float32
        )

        # --- Game state (set properly in reset()) ---
        self.snake = None
        self.food = None



    def get_state(self):
        head_x, head_y = self.snake.body[0]
        dir_x, dir_y = self.snake.direction

        # Helper to check What's in the cell directly ahead in each absolute direction?
        def is_dangerous(x, y):
            # Wall collision
            if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
                return 1.0
            # Self collision
            if (x, y) in self.snake.body[1:]:
                return 1.0
            return 0.0

        danger_up    = is_dangerous(head_x, head_y - 1)
        danger_down  = is_dangerous(head_x, head_y + 1)
        danger_left  = is_dangerous(head_x - 1, head_y)
        danger_right = is_dangerous(head_x + 1, head_y)

        # --- Food direction ---
        food_x, food_y = self.food.position
        food_up    = 1.0 if food_y < head_y else 0.0
        food_down  = 1.0 if food_y > head_y else 0.0
        food_left  = 1.0 if food_x < head_x else 0.0
        food_right = 1.0 if food_x > head_x else 0.0

        # --- Current direction (one-hot) ---
        #In pygame (0, -1) is up, (0, 1) is down, (-1, 0) is left, (1, 0) is right
        dir_up    = 1.0 if dir_y == -1 else 0.0
        dir_down  = 1.0 if dir_y ==  1 else 0.0
        dir_left  = 1.0 if dir_x == -1 else 0.0
        dir_right = 1.0 if dir_x ==  1 else 0.0

        #np array with float32 for Pytorch compatibility
        return np.array([
            danger_up, danger_down, danger_left, danger_right,
            food_up, food_down, food_left, food_right,
            dir_up, dir_down, dir_left, dir_right
        ], dtype=np.float32)


    #To reset the environment to an initial state 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # handles seeding for reproducibility

        # Fresh game objects every episode
        self.snake = Snake()
        self.food = Food(self.snake.body)

        # Return initial state and empty info dict
        observation = self.get_state()
        return observation, {}


    def step(self, action):

        current_index = DIRECTIONS.index(self.snake.direction)

        #Going straight
        if action == 0:
            new_index = current_index

        #Turn right
        elif action == 1:
            new_index = (current_index + 1) % 4

        #Turn Left
        else:
            new_index = (current_index - 1) % 4


        self.snake.direction = DIRECTIONS[new_index]
        self.snake.move()

        reward = 0.0
        terminated = False

        #Negative reward if dead and positive if we found and ate food
        if self.snake.is_dead():
            reward = -10
            terminated = True
        elif self.snake.body[0] == self.food.position:
            reward = 10.0
            self.snake.grow()
            self.food = Food(self.snake.body)

        observation = self.get_state()

        info = {
            "score": len(self.snake.body) - 3,
            "won": len(self.snake.body) == GRID_SIZE * GRID_SIZE  # just a flag, no special reward
        }

        #return observation, reward, terminated, truncated, info
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

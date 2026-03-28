import random

import pygame

from utils.constants import CELL_SIZE, GRID_SIZE



class Food:

    def __init__(self, snake_body):
        self.position = self.spawn(snake_body)

    def spawn(self, snake_body):
        """
        Determine a random free position on the grid that is not occupied by the snake.

        Returns:
            tuple[int, int] | None: A free (x, y) position, or None if no free cells remain.
        """
        # Build a list of all free cells and pick one at random. This avoids an
        # unbounded loop when the snake fills the entire grid.
        occupied = set(snake_body)
        free_cells = [
            (x, y)
            for x in range(GRID_SIZE)
            for y in range(GRID_SIZE)
            if (x, y) not in occupied
        ]

        if not free_cells:
            # No free cells remain; signal this by returning None.
            return None

        return random.choice(free_cells)
            
    def draw(self, screen):
        RED = (200, 0, 0)
        # If no position is available (e.g., grid is full), do not draw food.
        if self.position is None:
            return
        x, y = self.position
        pygame.draw.rect(screen, RED, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
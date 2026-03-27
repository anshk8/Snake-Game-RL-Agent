import random

import pygame

from utils.constants import CELL_SIZE, GRID_SIZE



class Food:

    def __init__(self, snake_body):
        self.position = self.spawn(snake_body)

    def spawn(self, snake_body):
        # Keep generating random positions until one is not on the snake
        while True:
            pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if pos not in snake_body:
                return pos
            
    def draw(self, screen):
        RED = (200, 0, 0)
        x, y = self.position
        pygame.draw.rect(screen, RED, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
from utils.constants import WIDTH, HEIGHT, GRID_SIZE, CELL_SIZE
import pygame as pg

GREY = (40, 40, 40)   # subtle grid lines
WHITE = (255, 255, 255)

# Draws the grid lines on the screen
def draw_grid(screen: pg.Surface):
    for x in range(0, WIDTH, CELL_SIZE):        # vertical lines
        pg.draw.line(screen, GREY, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):       # horizontal lines
        pg.draw.line(screen, GREY, (0, y), (WIDTH, y))

def draw_border(screen):
    pg.draw.rect(screen, WHITE, (0, 0, WIDTH, HEIGHT), 2)  # 2px border

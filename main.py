import pygame as pg
import sys
from game.player import Snake
from utils.constants import WIDTH, HEIGHT
from game.board import draw_border, draw_grid

pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Snake Game")

def main():
    snake = Snake()
    clock = pg.time.Clock()

    #Reduces lag by only moving the snake on a timer event instead of every frame
    MOVE_EVENT = pg.USEREVENT + 1          # custom timer event
    pg.time.set_timer(MOVE_EVENT, 200)     # fire every 200ms

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    snake.direction = (0, -1)
                if event.key == pg.K_DOWN:
                    snake.direction = (0, 1)
                if event.key == pg.K_LEFT:
                    snake.direction = (-1, 0)
                if event.key == pg.K_RIGHT:
                    snake.direction = (1, 0)

            if event.type == MOVE_EVENT:   # only move on timer tick
                snake.move()

        screen.fill((0, 0, 0))
        draw_grid(screen)
        draw_border(screen)
        snake.draw(screen, pg)
        pg.display.update()
        clock.tick(60)                    

main()
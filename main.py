import pygame as pg
import sys
from game.food import Food
from game.player import Snake
from utils.constants import WIDTH, HEIGHT
from game.board import draw_border, draw_grid

pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Snake Game")


font = pg.font.SysFont("Arial", 24)

def draw_score(screen, score):
    text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(text, (10, 10))   # top left corner


def main():
    snake = Snake()
    clock = pg.time.Clock()
    food = Food(snake.body)

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

            # inside MOVE_EVENT block:
            if event.type == MOVE_EVENT:
                snake.move()

                if snake.is_dead():
                    snake = Snake()           # reset snake
                    food = Food(snake.body)   # reset food
                    print("Dead! Restarting...")  

                elif snake.body[0] == food.position:
                    snake.grow()
                    food = Food(snake.body)

        screen.fill((0, 0, 0))
        draw_grid(screen)
        draw_border(screen)
        food.draw(screen)
        snake.draw(screen)
        draw_score(screen, len(snake.body) - 3)  # Initial snake length is 3
        pg.display.update()
        clock.tick(60)                    

main()
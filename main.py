import pygame as pg
import sys
from utils.constants import WIDTH, HEIGHT


#NOTE: Did this project to also learn about pygame, comments explaining around

pg.init()

screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Snake Game")


def main():

    while True:
        #event.get() hecks for things like mouse clicks, keypresses, closing the window
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        screen.fill((0, 0, 0))

        #Pushes the drawn frame to your screen
        pg.display.update()

main()
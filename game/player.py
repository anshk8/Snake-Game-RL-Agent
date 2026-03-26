from utils.constants import CELL_SIZE
import pygame

class Snake:
    def __init__(self):

        #Start body at center (10, 10) and snake is 3 segments long, moving right
        self.body = [(10, 10), (9, 10), (8, 10)]  

        #Starts moving right
        self.direction = (1, 0)                 


    #Draw the snake on our screen
    def draw(self, screen: pygame.Surface):
        GREEN = (0, 200, 0)
        for segment in self.body:
            x, y = segment
            pygame.draw.rect(screen, GREEN, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    #Movement 
    def move(self):

        #Take current position of head and add direction to get new head position
        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)

        # add new head
        self.body.insert(0, new_head)  

        # remove tail
        self.body.pop()                
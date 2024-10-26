import pygame
import sys
import os

pygame.init()

screen_width, screen_height = 1920, 1080
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Drawing App")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
color = BLACK 


drawing = False 
brush_size = 10 

screen.fill(WHITE)

def quit():
    pygame.image.save(screen, 'temp_img.png')
    pygame.quit()
    os.system('python neural.py')
    sys.exit()
    
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:            
            quit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                quit()
            if event.key == pygame.K_c:
                screen.fill(WHITE)
            elif event.key == pygame.K_u:
                brush_size = min(brush_size + 1, 20) 
            elif event.key == pygame.K_d:
                brush_size = max(brush_size - 1, 1)   

    if drawing:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        pygame.draw.circle(screen, color, (mouse_x, mouse_y), brush_size)

    pygame.display.flip()

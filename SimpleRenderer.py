import pygame
from pygame.locals import *
from math import *

from OpenGL.GL import *
from OpenGL.GLU import *

class SimpleRenderer():
    def __init__(self, width, height):
        self.circleSides = 32
        self.width, self.height = 800, 600

    def init(self):
        pygame.init()
        pygame.display.set_mode( [self.width, self.height], DOUBLEBUF|OPENGL)
        glMatrixMode(GL_PROJECTION)
        glOrtho(0, 800, 0, 600, 0.0, -999.0);


    def clear(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    def drawCircle(self, cx, cy, r, color):
        glColor3f(color[0], color[1], color[2]);
        glBegin(GL_POLYGON)
        for i in range(self.circleSides):
            cosine= r * cos(i*2*pi/self.circleSides) + cx
            sine  = r * sin(i*2*pi/self.circleSides) + cy
            glVertex2f(cosine, sine)
        glEnd();

    def flipBuffer(self):
        pygame.display.flip()

    def captureInput(self):
        pygame.key.set_repeat(16, 16)

        events = pygame.event.get()
        ## Quit, UP, DOWN
        input = [False]*5
        for event in events:
            if event.type == pygame.QUIT:
                input[0] = True
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_ESCAPE]:
                input[0] = True
            if pressed[pygame.K_UP]:
                input[1] = True
            if pressed[pygame.K_DOWN]:
                input[2] = True
            if pressed[pygame.K_LEFT]:
                input[3] = True
            if pressed[pygame.K_RIGHT]:
                input[4] = True
        return input

    def time(self):
        return pygame.time.get_ticks()

    def wait(self, time):
        pygame.time.wait(time)


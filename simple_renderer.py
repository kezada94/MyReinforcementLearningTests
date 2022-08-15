import pygame
from pygame.locals import *
from math import *

from OpenGL.GL import *
from OpenGL.GLU import *

class SimpleRenderer():
    def __init__(self, viewportWidth, viewportHeight, headlessMode = False):
        self.viewportWidth, self.viewportHeight = viewportWidth, viewportHeight
        self.headlessMode = headlessMode

    def init(self):
        pygame.init()

        flags = DOUBLEBUF|OPENGL
        if self.headlessMode:
            flags |= HIDDEN
        
        self.screen = pygame.display.set_mode( [self.viewportWidth, self.viewportHeight], flags)
        glMatrixMode(GL_PROJECTION)
        glOrtho(-1, 1, -1, 1, 0.0, -999.0);


    def clear(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    def drawShape(self, vertices, color):
        glColor3f(color[0], color[1], color[2]);
        glBegin(GL_POLYGON)
        for vertex in vertices:
            glVertex2f(vertex[0], vertex[1])
        glEnd();

    def flipBuffer(self):
        pygame.display.flip()

    def captureInput(self):
        pygame.key.set_repeat(16, 16)

        events = pygame.event.get()
        ## Quit, UP, DOWN
        input = [False]*6
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
            if pressed[pygame.K_r]:
                input[5] = True
        return input

    def exportFrameAs3DArray(self):
        size = self.screen.get_size()   
        buffer = glReadPixels(0, 0, *size, GL_RGB, GL_UNSIGNED_BYTE)
        #print(type)
        screen_surf = pygame.image.fromstring(buffer, size, "RGB")
        #screen_surf = pygame.image.frombuffer(buffer, size, "RGB")
        #screen_surf = pygame.transform.flip(screen_surf,flip_x=False, flip_y=True)
        return pygame.surfarray.array3d(screen_surf)

    def time(self):
        return pygame.time.get_ticks()

    def wait(self, time):
        pygame.time.wait(time)
        
    def close(self):
        pygame.display.quit()
        pygame.quit()


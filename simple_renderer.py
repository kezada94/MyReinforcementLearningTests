import pygame
from pygame.locals import *
from math import *
import time as t

from OpenGL.GL import *
from OpenGL.GLU import *

class SimpleRenderer():
    def __init__(self, viewportWidth, viewportHeight, headlessMode = False):
        self.viewportWidth, self.viewportHeight = viewportWidth, viewportHeight
        self.headlessMode = headlessMode

    def _createPBO(self, w, h):
        pbo = glGenBuffers (1)
        glBindBuffer (GL_PIXEL_PACK_BUFFER, pbo)
        glBufferData(target = GL_PIXEL_PACK_BUFFER, size=w*h*4 + 144, data=None, usage=GL_STREAM_COPY);
        glBindBuffer (GL_PIXEL_PACK_BUFFER, 0)
        return pbo

    def init(self):
        pygame.init()

        flags = OPENGL | DOUBLEBUF
        if self.headlessMode:
            flags |= HIDDEN
        
        self.screen = pygame.display.set_mode( [self.viewportWidth, self.viewportHeight], flags)
        print(pygame.display.Info())
        glMatrixMode(GL_PROJECTION)
        glOrtho(-1, 1, -1, 1, 0.0, -999.0);
        self.pbo = self._createPBO(self.viewportWidth, self.viewportHeight)


    def clear(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    def drawShape(self, vertices, color):
        
        glColor3f(color[0], color[1], color[2]);
        glBegin(GL_POLYGON)
        for vertex in vertices:
            glVertex2f(*vertex)
        glEnd();
        #glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_WRITE)
        #glBindBuffer (GL_PIXEL_PACK_BUFFER, 0)
        #glUnmapBuffer(GL_PIXEL_PACK_BUFFER)

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
        
        glReadBuffer(GL_FRONT);
        #glBufferData()
        glBindBuffer (GL_PIXEL_PACK_BUFFER, self.pbo)
        #glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY)

        buffer = glReadPixels(0, 0, self.viewportWidth, self.viewportHeight, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE)
        
        glBindBuffer (GL_PIXEL_PACK_BUFFER, 0)
        #glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY)
        #print(self.pbo1)
        #print(buffer)
        #print(type)
        
        glDrawBuffer(GL_BACK);
        #s = self.screen.copy()
        #s = pygame.PixelArray(s)
        #print(s)
        #exit()
        #screen_surf = pygame.image.fromstring(buffer, size, "RGB")
        #screen_surf = pygame.image.frombuffer(buffer, size, "RGB")
        #screen_surf = pygame.transform.flip(s,flip_x=False, flip_y=True)
        #return pygame.surfarray.array3d(screen_surf)
        return 2

    def time(self):
        return t.perf_counter()

    def wait(self, time):
        pygame.time.wait(time)
        
    def close(self):
        pygame.display.quit()
        pygame.quit()


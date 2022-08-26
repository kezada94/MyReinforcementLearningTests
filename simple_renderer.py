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

    def _createFBO(self, w, h):
        fbo = glGenFramebuffers (1)
        glBindFramebuffer (GL_FRAMEBUFFER, fbo)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
        
        return fbo

    def _createPBO(self, w, h):
        pbo = glGenBuffers (1)
        glBindBuffer (GL_PIXEL_PACK_BUFFER, pbo)
        glBufferData(target = GL_PIXEL_PACK_BUFFER, size=w*h*3, data=None, usage=GL_STREAM_COPY);
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
        self.fbo = self._createFBO(self.viewportWidth, self.viewportHeight)
        self.pbo = self._createPBO(self.viewportWidth, self.viewportHeight)

    def clear(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    def drawShape(self, vertices, color):
        
        glColor3f(color[0], color[1], color[2]);
        glBegin(GL_POLYGON)
        for vertex in vertices:
            glVertex2f(*vertex)
        glEnd();

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        self.clear()
        glColor3f(color[0], color[1], color[2]);
        glBegin(GL_POLYGON)
        for vertex in vertices:
            glVertex2f(*vertex)
        glEnd();
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

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
        glBindFramebuffer (GL_FRAMEBUFFER, self.fbo)

        buffer = glReadPixels(0, 0, self.viewportWidth, self.viewportHeight, GL_RGB, GL_UNSIGNED_BYTE)
        glBindFramebuffer (GL_FRAMEBUFFER, 0)

        screen_surf = pygame.image.fromstring(buffer, (self.viewportWidth, self.viewportHeight), "RGB")
        #screen_surf = pygame.image.frombuffer(buffer, size, "RGB")
        #screen_surf = pygame.transform.flip(s,flip_x=False, flip_y=True)
        return pygame.surfarray.array3d(screen_surf)

    def time(self):
        return t.perf_counter()

    def wait(self, time):
        pygame.time.wait(time)
        
    def close(self):
        pygame.display.quit()
        pygame.quit()


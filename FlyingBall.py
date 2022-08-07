from SimpleRenderer import SimpleRenderer
import random
from math import *
from PIL import Image
import numpy as np

CIRCLE_VERTICES = 32

class Ball():
    def __init__(self, radius, color):
        self.radius = radius
        self.position = np.array([0,0]);
        self.color = color
        self.velocity = np.array([0,0]);
        self.genModel()

    def genModel(self):
        #self.vertices = np.ndarray(CIRCLE_VERTICES*2)
        R = np.repeat(self.radius, CIRCLE_VERTICES)
        D = np.arange(CIRCLE_VERTICES)*2*pi/CIRCLE_VERTICES
        X = R * np.cos(D)
        Y = R * np.sin(D)
        self.vertices = np.stack((X, Y), 1)

    def toWorldSpace(self):
        return self.vertices+self.position
        # render

class Region():
    #                   vec2        vec2
    def __init__(self, topLeft, bottomRight):
        self.topLeft = topLeft
        self.bottomRight = bottomRight

    def isIn(self, position):
        if position[0]<self.topLeft[0] or position[0]>self.bottomRight[0] or position[1]>self.topLeft[1] or position[1]<self.bottomRight[1]:
            return False
        return True

class Camera():
    def __init__(self, position, aspectRatio, height):
        self.aspectRatio = aspectRatio
        self.position = position
        self.height = height

    def toViewSpace(self, vertices):
        transformer = np.array([1/self.height*self.aspectRatio, 1/self.height])
        newVertices = vertices*transformer
        #this can be done much faster with vectorization in numpy 
        #for i, vertex in enumerate(vertices):
        #    newVertices[i] = [vertex[2*i]/(self.height*self.aspectRatio), vertex[2*i+1]/(self.height)]
        return newVertices

class FlyingBallGame():
    def __init__(self, viewportWidth, viewportHeight, cameraHeight, gravity=np.array([0, -0.1]), playerRadius=5, playerJumpVelocity=np.array([0.0, 3]), enemyRadius=10, enemyVelocity=np.array([-1, 0.0]), maxEnemies=10, enemyProb=0.0005, gameSpeed=1):
        self.r = SimpleRenderer(viewportWidth, viewportHeight)
        self.aspectRatio = viewportWidth/viewportHeight
        self.camera = Camera(np.array([0, 0]), self.aspectRatio, cameraHeight)
        self.r.init()
        self.gameShouldQuit = False
        self.lastTime = 0
        self.viewportWidth, self.viewportHeight = viewportWidth, viewportHeight
        self.gameSpeed = gameSpeed

        self.player = Ball(playerRadius, [255, 0, 0])
        self.playerJumpVelocity = playerJumpVelocity
        self.enemyVelocity = enemyVelocity
        self.enemyRadius = enemyRadius
        self.enemies = []
        self.maxEnemies = maxEnemies
        self.enemyProb = enemyProb
        self.gravity = gravity

        self.spawnRegion = Region(np.array([100,100]), np.array([100,-100]))
        self.despawnRegion = Region(np.array([-102,100]), np.array([-100,-100]))
        self.aliveRegion = Region(np.array([-100,100]), np.array([100,-100]))
        self.reset()

    def reset(self):
        self.player.position = np.array([0.0,0.0])
        self.player.velocity = np.array([0.0,0.0])
        self.enemies = []

    def updatePhysics(self):
        self.player.velocity += self.gravity*self.gameSpeed
        # print(self.playerJumpVelocity)
        self.player.position += self.player.velocity*self.gameSpeed

        for enemy in self.enemies:
            enemy.position += enemy.velocity*self.gameSpeed

    def playerJump(self):
        # print("jump!")
        self.player.velocity = self.playerJumpVelocity.copy()

        # if self.player.position[1] < self.player.radius:
        #    self.player.position[1] = self.player.radius

    def processInput(self, input):
        if input[0]:
            self.gameShouldQuit = True
        if input[1]:
            self.playerJump()
        if input[3]:
            self.camera.height += 1
        if input[4]:
            # this cant be zero
            self.camera.height -= 1


    def drawScene(self):
        worldVertices = self.player.toWorldSpace()
        viewVertices = self.camera.toViewSpace(worldVertices)
        self.r.drawShape(viewVertices, self.player.color)
        
        b = Ball(5, np.array([0,0,255]))
        b.position = np.array([0,50])
        wb = b.toWorldSpace()
        vb = self.camera.toViewSpace(wb)
        self.r.drawShape(vb, b.color)

        for enemy in self.enemies:
            worldVerticesE = enemy.toWorldSpace()
            viewVerticesE = self.camera.toViewSpace(worldVerticesE)
            self.r.drawShape(viewVerticesE, enemy.color)


    def spawnEnemy(self):
        if len(self.enemies) < self.maxEnemies and random.random() < self.enemyProb:
            enemy = Ball(self.enemyRadius, [0, 255, 0])
            x = random.uniform(self.spawnRegion.topLeft[0], self.spawnRegion.bottomRight[0])
            y = random.uniform(self.spawnRegion.bottomRight[1], self.spawnRegion.topLeft[1])
            enemy.position = np.array([x, y])
            enemy.velocity = self.enemyVelocity
            self.enemies.append(enemy)

    def distance(self, pos1, pos2):
        return sqrt(abs(pos1[0]-pos2[0])**2 + abs(pos1[1]-pos2[1])**2)

    def checkCollision(self):
        delete = []
        for enemy in self.enemies:
            if self.despawnRegion.isIn(enemy.position):
                delete.append(enemy)
            if (self.distance(self.player.position, enemy.position) <= max(self.player.radius, enemy.radius)):
                self.gameShouldQuit = True
        if not self.aliveRegion.isIn(self.player.position):
            self.gameShouldQuit = True

        for d in delete[::-1]:
            self.enemies.remove(d)

    def mainLoop(self, updateFreq=16.66):
        self.updateFreq = updateFreq
        self.lastTime = self.r.time()
        while not self.gameShouldQuit:
            now = self.r.time()
            self.deltaTime = now - self.lastTime
            if (self.deltaTime < updateFreq):
                continue
            print(1/self.deltaTime*1000)
            self.lastTime = now
            self.r.clear()
            input = self.r.captureInput()
            self.processInput(input)
            self.spawnEnemy()
            self.updatePhysics()
            self.checkCollision()
            self.drawScene()
            self.r.flipBuffer()

    
        # capture input


game = FlyingBallGame(800, 800, 100, playerRadius=5,
                    enemyRadius=5, enemyProb=0.004, gameSpeed=1)
game.mainLoop()
frame = game.r.exportFrameAs3DArray()
im = Image.fromarray(np.transpose(frame[::2, ::2], (1, 0, 2)), 'RGB')
im.save("lastFrame.png")

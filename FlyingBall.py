from SimpleRenderer import SimpleRenderer
import random
from math import *
from PIL import Image
import numpy as np

class Ball():
    def __init__(self, radius, color):
        self.radius = radius
        self.position = np.array([0, 0])
        self.color = color
        self.velocity = np.array([0, 0])


class FlyingBallGame():
    def __init__(self, width, height, gravity = np.array([0, -0.001]), playerRadius = 5, playerJumpVelocity = np.array([0, 0.3]), enemyRadius = 10, enemyVelocity = np.array([-0.2, 0]), maxEnemies= 10, enemyProb = 0.0005):
        self.r = SimpleRenderer(width, height)
        self.r.init()
        self.gameShouldQuit = False
        self.lastTime = self.r.time()
        self.width, self.height = width, height

        self.player = Ball(playerRadius, [255, 0, 0])
        self.playerJumpVelocity = playerJumpVelocity
        self.enemyVelocity = enemyVelocity
        self.enemyRadius = enemyRadius
        self.enemies = []
        self.maxEnemies = maxEnemies
        self.enemyProb = enemyProb
        self.aspectRatio = width/height
        self.gravity = gravity
        self.reset()

    def reset(self):
        self.player.position = [self.width/2, self.height/2]
        self.player.velocity = [0, 0]
        self.enemies = []


    def updatePhysics(self):
        self.player.velocity += self.gravity*self.deltaTime
        print(self.playerJumpVelocity)
        
        self.player.position += self.player.velocity*self.deltaTime

        for enemy in self.enemies:
            enemy.position += enemy.velocity*self.deltaTime

    def playerJump(self):
        print("jump!")
        self.player.velocity = self.playerJumpVelocity.copy()
        
        #if self.player.position[1] < self.player.radius:
        #    self.player.position[1] = self.player.radius

    def processInput(self, input):
        if input[0]:
            self.gameShouldQuit = True
        if input[1]:
            self.playerJump()
        

    def drawScene(self):
        self.r.drawCircle(self.player.position[0], self.player.position[1], self.player.radius, self.player.color)
        for enemy in self.enemies:
            self.r.drawCircle(enemy.position[0], enemy.position[1], enemy.radius, enemy.color)


    def spawnEnemy(self):
        if len(self.enemies) < self.maxEnemies and random.random()<self.enemyProb:
            enemy = Ball(self.enemyRadius, [0, 255, 0])
            y = random.randint(enemy.radius, self.height-enemy.radius)
            enemy.position = [self.width, y]
            enemy.velocity = self.enemyVelocity
            self.enemies.append(enemy)

    def distance(self, x1, y1, x2, y2):
        return sqrt(abs(x1-x2)**2 + abs(y1-y2)**2)

    def checkCollision(self):
        delete = []
        for enemy in self.enemies:
            if enemy.position[0]<0:
                delete.append(enemy)
            if (self.distance(self.player.position[0], self.player.position[1], enemy.position[0], enemy.position[1]) <= max(self.player.radius, enemy.radius)):
                self.gameShouldQuit = True
        if self.player.position[1] <= 0 or self.player.position[1] >= self.height:
            self.gameShouldQuit = True

        for d in delete[::-1]:
            self.enemies.remove(d)
        
    def mainLoop(self, updateFreq=1./60.):
        self.updateFreq = updateFreq
        while not self.gameShouldQuit:
            now = self.r.time()
            self.deltaTime = now - self.lastTime
            if (self.deltaTime < updateFreq):
                continue
            self.lastTime = now
            self.r.clear()
            input = self.r.captureInput()
            self.processInput(input)
            self.drawScene()
            self.spawnEnemy()
            self.updatePhysics()
            self.checkCollision()
            self.r.flipBuffer()

        #render
        #capture input




game = FlyingBallGame(800, 600, playerRadius = 10, enemyRadius=10, enemyProb=0.0004)
game.mainLoop()
frame = game.r.exportFrameAs3DArray()
im = Image.fromarray(np.transpose(frame[::2, ::2], (1, 0, 2)), 'RGB')
im.save("lastFrame.png")

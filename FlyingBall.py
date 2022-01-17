from SimpleRenderer import SimpleRenderer
import random
from math import *
class FlyingBallGame():
    def __init__(self, width, height, gravity=0, playerRadius = 5, playerMovementSpeed = 7, enemyMovementSpeed = 0.2, maxEnemies= 10, enemyProb = 0.005):
        self.r = SimpleRenderer(width, height)
        self.r.init()

        self.lastTime = self.r.time()
        self.width, self.height = width, height
        self.playerMovementSpeed = playerMovementSpeed
        self.enemyMovementSpeed = enemyMovementSpeed
        self.playerRadius = playerRadius
        self.enemies = []
        self.enemyRadius = 5
        self.maxEnemies = maxEnemies
        self.enemyProb = enemyProb
        self.reset()
        self.gameShouldQuit = False

    def reset(self):
        self.playerPosition = [self.width/2, self.height/2]
        self.enemies = []

    def movePlayerUp(self):
        self.playerPosition[1] += self.playerMovementSpeed*self.deltaTime
        if self.playerPosition[1]>self.height-self.playerRadius:
            self.playerPosition[1] = self.height - self.playerRadius

    def movePlayerDown(self):
        self.playerPosition[1] -= self.playerMovementSpeed*self.deltaTime
        if self.playerPosition[1] < self.playerRadius:
            self.playerPosition[1] = self.playerRadius

    def processInput(self, input):
        if input[0]:
            self.gameShouldQuit = True
        if input[1]:
            self.movePlayerUp()
        if input[2]:
            self.movePlayerDown()

    def drawScene(self):
        self.r.drawCircle(self.playerPosition[0], self.playerPosition[1], self.playerRadius, [255, 0, 0])
        for enemy in self.enemies:
            self.r.drawCircle(enemy[0], enemy[1], enemy[2], [0,255,0])


    def shouldSpawn(self):
        return len(self.enemies) < self.maxEnemies and random.random()<self.enemyProb

    def spawnEnemy(self):
        if self.shouldSpawn():
            y = random.randint(self.enemyRadius, self.height-self.enemyRadius)
            self.enemies.append( [self.width, y, self.enemyRadius] )

    def step(self):
        i = 0
        while i < len(self.enemies):
            self.enemies[i][0] -= self.enemyMovementSpeed*self.deltaTime
            if self.enemies[i][0] < 0:
                self.enemies.remove(self.enemies[i])
            else:
                i+=1

    def distance(self, x1, y1, x2, y2):
        return sqrt(abs(x1-x2)**2 + abs(y1-y2)**2)

    def checkCollision(self):
        for enemy in self.enemies:
            if (self.distance(self.playerPosition[0], self.playerPosition[1], enemy[0], enemy[1]) <= max(self.playerRadius, enemy[2])):
                self.gameShouldQuit = True


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
            self.step()
            self.checkCollision()
            self.r.flipBuffer()

        #render
        #capture input




game = FlyingBallGame(800,600, playerRadius = 25)
game.mainLoop()

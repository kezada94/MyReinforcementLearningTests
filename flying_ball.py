from simple_renderer import SimpleRenderer
import random
from math import *
import numpy as np
import enum
CIRCLE_VERTICES = 16

class Actions(enum.Enum):
    Noot = 0
    Jump = 1

class Ball():
    def __init__(self, radius, color):
        self.radius = radius
        self.position = np.array([0,0]);
        self.color = color
        self.velocity = np.array([0,0]);
        self.isAlive = False
        self.genModel()

    def genModel(self):
        #self.vertices = np.ndarray(CIRCLE_VERTICES*2)
        R = np.repeat(self.radius, CIRCLE_VERTICES)
        D = np.arange(CIRCLE_VERTICES)*2*pi/CIRCLE_VERTICES
        X = R * np.cos(D)
        Y = R * np.sin(D)
        self.vertices = np.stack((X, Y), 1)
        print(self.vertices.shape)

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
        transformer = np.array([1/self.height*2*self.aspectRatio, 1/self.height*2])
        newVertices = vertices*transformer
        #this can be done much faster with vectorization in numpy 
        #for i, vertex in enumerate(vertices):
        #    newVertices[i] = [vertex[2*i]/(self.height*self.aspectRatio), vertex[2*i+1]/(self.height)]
        return newVertices

class FlyingBallGame():
    def __init__(self, viewportWidth, viewportHeight, cameraHeight, headlessMode = False, gravity=np.array([0, -0.1]), playerRadius=5, playerJumpVelocity=np.array([0.0, 3]), enemyRadius=10, enemyVelocity=np.array([-1, 0.0]), maxEnemies=10, enemyProb=0.0005, gameSpeed=1, seed=0):
        self.r = SimpleRenderer(viewportWidth, viewportHeight, headlessMode=headlessMode)
        self.r.init()
        random.seed(seed)
    
        self.aspectRatio = viewportWidth/viewportHeight
        self.camera = Camera(np.array([0, 0]), self.aspectRatio, cameraHeight)
        self.gameShouldQuit = False
        self.gameShouldReset = False
        self.lastTime = 0
        self.viewportWidth, self.viewportHeight = viewportWidth, viewportHeight
        self.gameSpeed = gameSpeed
        self.score = 0
        self.framesSinceReset = 0
        self.player = Ball(playerRadius, [255, 0, 0])
        self.player.isAlive = True
        self.playerJumpVelocity = playerJumpVelocity
        self.enemyVelocity = enemyVelocity
        self.enemyRadius = enemyRadius
        self.maxEnemies = maxEnemies
        self.enemies = [Ball(enemyRadius, [0, 255, 0]) for x in range(self.maxEnemies)]
        self.enemiesAlive = 0
        self.enemyProb = enemyProb
        self.gravity = gravity
        self.spawnRegion = Region(np.array([100,100]), np.array([100,-100]))
        self.despawnRegion = Region(np.array([-102,100]), np.array([-100,-100]))
        self.aliveRegion = Region(np.array([-100,100]), np.array([100,-100]))
        self.outliers = 
        self.reset()

    def reset(self):
        self.player.position = np.array([0.0, 0.0])
        self.player.velocity = np.array([0.0, 0.0])
        self.player.isAlive = True

        #could be only for alive enemies
        for enemy in self.enemies:
            enemy.isAlive = False
        self.score = 0
        self.framesSinceReset = 0
        self.gameShouldReset=False
        self.enemiesAlive = 0

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
        actions = []
        if input[0]:
            self.gameShouldQuit = True
        if input[1]:
            actions.append(Actions.Jump)
            #self.playerJump()
        if input[3]:
            self.camera.height += 1
        if input[4]:
            # this cant be zero
            self.camera.height -= 1
            assert self.camera.height>0
        if input[5]:
            self.gameShouldReset = True
        return actions

    def drawScene(self):
        worldVertices = self.player.toWorldSpace()
        viewVertices = self.camera.toViewSpace(worldVertices)
        self.r.drawShape(viewVertices, self.player.color)
        
        # b = Ball(5, np.array([0,0,255]))
        # b.position = np.array([0,50])
        # wb = b.toWorldSpace()
        # vb = self.camera.toViewSpace(wb)
        # self.r.drawShape(vb, b.color)

        for enemy in self.enemies:
            if enemy.isAlive:
                worldVerticesE = enemy.toWorldSpace()
                viewVerticesE = self.camera.toViewSpace(worldVerticesE)
                self.r.drawShape(viewVerticesE, enemy.color)

    def addOutlierBall(self, position, color, radius):


    def spawnEnemy(self):
        if self.enemiesAlive < self.maxEnemies and random.random() < self.enemyProb:
            #print("1", [e.isAlive for e in self.enemies])

            #print("Spawning enemy.")
            enemy = None
            for e in self.enemies:
                if not e.isAlive:
                    enemy = e
                    break
            assert enemy is not None
            #print("2", [e.isAlive for e in self.enemies])

            x = random.uniform(self.spawnRegion.topLeft[0], self.spawnRegion.bottomRight[0])
            y = random.uniform(self.spawnRegion.bottomRight[1], self.spawnRegion.topLeft[1])
            enemy.position = np.array([x, y])
            enemy.velocity = self.enemyVelocity
            enemy.isAlive = True
            self.enemiesAlive+=1

            #self.enemies.append(enemy)

    def distance(self, pos1, pos2):
        return sqrt(abs(pos1[0]-pos2[0])**2 + abs(pos1[1]-pos2[1])**2)

    def checkCollision(self):
        #delete = []
        for enemy in self.enemies:
            if not enemy.isAlive:
                continue
            if self.despawnRegion.isIn(enemy.position):
                self.despawnEnemy(enemy)
            if (self.distance(self.player.position, enemy.position) <= max(self.player.radius, enemy.radius)):
                #print("player out of collision")
                self.player.isAlive = False
                #self.gameShouldReset = True
        if not self.aliveRegion.isIn(self.player.position):
            #print("player out of region")
            self.player.isAlive = False
            #self.gameShouldReset = True

        #for d in delete[::-1]:
        #    self.enemies.remove(d)
    def despawnEnemy(self, enemy):
        enemy.isAlive = False
        self.enemiesAlive-=1
        assert self.enemiesAlive>-1

    ## maxFPS is the minimum ms between frames
    #def mainLoop(self, maxFPS=16.00):
    def mainLoop(self, maxFPS=0):
        self.maxFPS = maxFPS
        self.lastTime = self.r.time()
        self.acctime = 0
        while not self.gameShouldQuit:
            now = self.r.time()
            a=self.r.exportFrameAs3DArray()

            self.deltaTime = now - self.lastTime
            if (maxFPS != 0 and 1/self.deltaTime > maxFPS):
                #print(1/self.deltaTime)
                continue
            self.acctime += self.deltaTime

            if (self.acctime > 1 and self.deltaTime!=0):
                #continue
                print(1/self.deltaTime)
                self.acctime = 0
            self.lastTime = now
            keysPressed = self.r.captureInput()
            actions = self.processInput(keysPressed)
            self.step(actions)
            self.render()
            if self.gameShouldReset:
                self.reset()

    def close(self):
        self.r.close()

    def impartLogic(self):
        if not self.player.isAlive:
            self.gameShouldReset = True

    def updateScore(self):
        if self.player.isAlive:
            self.framesSinceReset += 1
            if self.framesSinceReset == 60:
                self.framesSinceReset = 0
                self.score += 1
                #print(self.score)

    def performActions(self, actionsVector):
        for action in actionsVector:
            if action == Actions.Jump:
                self.playerJump()

    def step(self, actionsVector):
        if not self.player.isAlive:
            return
        self.spawnEnemy()
        self.performActions(actionsVector)
        self.updatePhysics()
        self.checkCollision()
        #self.impartLogic()
        self.updateScore()
        
    def render(self):
        self.r.clear()
        self.drawScene()
        self.r.flipBuffer()

if __name__ == "__main__":

    from PIL import Image
    game = FlyingBallGame(800, 600, 200, headlessMode=False, playerRadius=5,
                    enemyRadius=5, enemyProb=0.0, gameSpeed=1,
                    enemyVelocity=np.array([-1, 0.0]), seed=0, gravity=[0,0])
    game.mainLoop()
    frame = game.r.exportFrameAs3DArray()
    game.close()
    im = Image.fromarray(np.transpose(frame[::2, ::2], (1, 0, 2)), 'RGB')
    im.save("lastFrame.png")

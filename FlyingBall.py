import pygame as pg

class FlyingBallGame():
    def __init__(self, width, height, gravity=0, playerRadius = 5, playerMovementSpeed = 5, enemyMovementSpeed = 4):
        pg.init()
        self.lastTime = pg.time.get_ticks()
        self.width, self.height = width, height
        self.playerMovementSpeed = playerMovementSpeed
        self.enemyMovementSpeed = enemyMovementSpeed
        self.playerRadius = playerRadius
        self.screen = pg.display.set_mode([width, height])
        self.surface = pg.Surface(self.screen.get_size(), pg.SRCALPHA, 32)

        self.enemies = []
        self.reset()
        self.gameShouldQuit = False

    def reset(self):
        self.screen.fill((50, 50, 50))
        self.playerPosition = [self.width/2, self.height/2]
        self.enemies = []

    #step(deltaTime = 1./60.):

    def movePlayerUp(self):
        self.playerPosition[1] += self.playerMovementSpeed*self.updateFreq
        if self.playerPosition[1]>self.height:
            self.playerPosition[1] = self.height

    def movePlayerDown(self):
        self.playerPosition[0] += self.playerMovementSpeed*self.updateFreq
        if self.playerPosition[0]>self.width:
            self.playerPosition[0] = self.width

    def captureInput(self):

        events = pg.event.get()
        ## Quit, UP, DOWN
        input = [False, False, False]
        for event in events:
            if event.type == pg.QUIT:
                input[0] = True
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE or event.unicode == "q":
                    input[0] = True
                if event.key == pg.K_UP:
                    input[1] = True
                if event.key == pg.K_DOWN:
                    input[2] = True
        return input

    def processInput(self, input):
        if input[0]:
            self.gameShouldQuit = True
        if input[1]:
            self.movePlayerUp()
        if input[2]:
            self.movePlayerDown()

    def render(self):
        pg.draw.circle(self.surface, (255, 0, 0), self.playerPosition, self.playerRadius)
        
    def mainLoop(self, updateFreq=1./60.):
        self.updateFreq = updateFreq
        while not self.gameShouldQuit:
            deltaTime = pg.time.get_ticks() - self.lastTime
            if deltaTime < self.updateFreq:
                pg.time.wait(self.updateFreq-deltaTime)
            self.lastTime = pg.time.get_ticks()
            input = self.captureInput()
            self.processInput(input)
            self.screen.fill((50, 50, 50))
            self.render()
            pg.display.flip()

            
        #render
        #capture input
        #pg.display.flip()




game = FlyingBallGame(800,600, playerRadius = 100)
game.mainLoop()

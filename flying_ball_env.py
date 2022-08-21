from cv2 import ROTATE_90_COUNTERCLOCKWISE
import flying_ball as fb
import numpy as np
import gym
from gym import spaces

class FlyingBallGym(gym.Env):
    def __init__(self, maxEnemies: int = 5, headless: bool=False, target_fps: int = 60):
        width = 800
        height = 600
        self.action_space = spaces.Discrete(2);
        self.observation_space = (width, height, 3)
        self.game = fb.FlyingBallGame(width, height, 200, playerRadius=10,
                    enemyRadius=5, enemyProb=0.004, gameSpeed=1,
                    enemyVelocity=np.array([-1, 0.0]), headlessMode=headless)
        self.lastScore=0

    def _actionSpaceToAction(self, action_number: int):
        #return FlyingBall.Actions(action_number)
        if action_number == 0:
            return fb.Actions.Noot
        if action_number == 1:
            return fb.Actions.Jump

    def step(self, action: int):
        terminated = False
        truncated = False
        ## Step the simulation
        a = self._actionSpaceToAction(action)
        self.game.step([a])

        self.game.render()

        ## get State
        state = self._getObs()
        reward=0
        if not self.game.player.isAlive:
            reward = -1
            terminated = True
            self.lastScore=0
        else:
            if self.game.score != self.lastScore:
                self.lastScore = self.game.score
                reward = 1
        info = self._getInfo()
        return state, reward, terminated, truncated, info

    def _getInfo(self):
        info = {"playerPosition":self.game.player.position, "totalScore":self.game.score, "nEnemiesAlive" : self.game.enemiesAlive}
        return info

    def _getObs(self):
        return self.game.r.exportFrameAs3DArray()

    def reset(self):
        #super().reset()
        self.game.reset()
        self.game.render()
        state = self._getObs()
        info = self._getInfo()
        return state, info

    def render(self):
        self.game.render()

    def close(self):
        self.game.close()
        return



if __name__ == "__main__":
    game = FlyingBallGym()
    import env_transformations as t
    #                                         WIDTH HEIGHT
    game = t.TransformStateWrap(game, dstSize=(12, 84))
    game = t.FrameSkipWrap(game, framesToSkip=8)
    game = t.StackFramesWrap(game, framesToStack=4)

    import cv2
    g, info = game.reset()
    print(info)
    g,_,_,_,info = game.step(0)
    print(info)
    g,_,_,_,info = game.step(0)
    print(info)
    g,_,_,_,info = game.step(0)
    print(info)

    print(g.shape)

    for i in range(1888):
        g, _, end,_, info = game.step(0)    
        if end:
            g1=cv2.rotate(g[0], rotateCode=ROTATE_90_COUNTERCLOCKWISE)
            g2=cv2.rotate(g[1], rotateCode=ROTATE_90_COUNTERCLOCKWISE)
            g3=cv2.rotate(g[2], rotateCode=ROTATE_90_COUNTERCLOCKWISE)
            g4=cv2.rotate(g[3], rotateCode=ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite("lastFrame1.jpg", g1)
            cv2.imwrite("lastFrame2.jpg", g2)
            cv2.imwrite("lastFrame3.jpg", g3)
            cv2.imwrite("lastFrame4.jpg", g4)
            #game.reset()
            exit()
        #print(info)
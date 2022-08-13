import flying_ball as fb
import numpy as np
import gym
from gym import spaces

class FlyingBallGym(gym.Env):
    def __init__(self, maxEnemies: int = 5, headless: bool=False, target_fps: int = 60):
        self.action_space = spaces.Discrete(2);
        self.game = fb.FlyingBallGame(800, 800, 200, playerRadius=5,
                    enemyRadius=5, enemyProb=0.004, gameSpeed=1,
                    enemyVelocity=np.array([-1, 0.0]))
        self.lastScore=0

    def _actionSpaceToAction(self, action_number: int):
        #return FlyingBall.Actions(action_number)
        if action_number == 0:
            return fb.Actions.Noot
        if action_number == 1:
            return fb.Actions.Jump

    def step(self, action: int):
        end = False
        ## Step the simulation
        a = self._actionSpaceToAction(action)
        self.game.step([a])

        ## get State
        state = self._getObs()
        reward=0
        if not self.game.player.isAlive:
            reward = -1
            end = True
            self.lastScore=0
        else:
            if self.game.score != self.lastScore:
                self.lastScore = self.game.score
                reward = 1
        info = self._getInfo()
        return state, reward, end, info

    def _getInfo(self):
        info = {"playerPosition":self.game.player.position, "totalScore":self.game.score, "nEnemiesAlive" : self.game.enemiesAlive}
        return info

    def _getObs(self):
        return self.game.r.exportFrameAs3DArray()

    def reset(self):
        #super().reset()
        self.game.reset()
        state = self._getObs()
        info = self._getInfo()
        return state, info

    def render(self):
        self.game.render()
    def close(self):
        self.game.close()
        return


game = FlyingBallGym()
import env_transformations as t

game = t.FrameSkipWrap(game, framesToSkip=0)


_,_,_,info = game.step(1)

for i in range(1888):
    _,_,end,info = game.step(0)    
    if end:
        game.reset()
    game.render()
    print(info)
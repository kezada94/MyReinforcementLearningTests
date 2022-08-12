import gym

class FrameSkipWrap(gym.Wrapper):
    def __init__(self, env, framesToSkip = 4):
        super().__init__(env)
        self.env = env
        self.framesToSkip = framesToSkip    
    def step(self, action):
        out = self.env.step(action)
        for i in range(self.framesToSkip-1):
            ## Note that the action is repeated, is this desirable?
            self.env.step(action)
        return out

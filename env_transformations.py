import gym

## Only frameskip as humans also sample their world in a somewhat fixed interval
## whatever they miss its not maxed or averaged
class FrameSkipWrap(gym.Wrapper):
    def __init__(self, env, framesToSkip = 4):
        super().__init__(env)
        self.env = env
        self.framesToSkip = framesToSkip    
    def step(self, action):
        istate, ir, idone, iinfo = self.env.step(action)
        for i in range(self.framesToSkip-1):
            if idone:
                break

            _, rr, idone, _ = self.env.step(action) ## Note that the action is repeated, is this desirable?
            r += rr

        if idone:
            ir=-1
        return (istate, ir, idone, iinfo)

class StackFramesWrap(gym.Wrapper):
    def __init__(self, env, framesToStack = 4):
        super().__init__(env)
        self.env = env
        self.framesToStack = framesToStack    

    def step(self, action):
        out = self.env.step(action)
        for i in range(self.framesToSkip-1):
            ## Note that the action is repeated, is this desirable?
            self.env.step(action)
        return out

    def reset(self, action):
        out = self.env.step(action)
        for i in range(self.framesToSkip-1):
            ## Note that the action is repeated, is this desirable?
            self.env.step(action)
        return out

class TransformStateWrap(gym.Wrapper):
    def __init__(self, env, framesToStack = 4):
        super().__init__(env)
        self.env = env
        self.framesToStack = framesToStack    

    def step(self, action):
        out = self.env.step(action)
        for i in range(self.framesToSkip-1):
            ## Note that the action is repeated, is this desirable?
            self.env.step(action)
        return out

    def reset(self, action):
        out = self.env.step(action)
        for i in range(self.framesToSkip-1):
            ## Note that the action is repeated, is this desirable?
            self.env.step(action)
        return out
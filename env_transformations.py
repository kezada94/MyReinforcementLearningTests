import gym

## Only frameskip as humans also sample their world in a somewhat fixed interval
## whatever they miss its not maxed or averaged
class FrameSkipWrap(gym.Wrapper):
    def __init__(self, env, framesToSkip = 4):
        super().__init__(env, new_step_api=True)
        self.env = env
        self.framesToSkip = framesToSkip    

    def step(self, action):
        istate, ir, iterminated, truncated, iinfo = self.env.step(action)
        for i in range(self.framesToSkip-1):
            if iterminated:
                break

            _, rr, iterminated, _, _ = self.env.step(action) ## Note that the action is repeated, is this desirable?
            r += rr

        if iterminated:
            ir=-1
        return (istate, ir, iterminated, truncated, iinfo)

class StackFramesWrap(gym.Wrapper):
    def __init__(self, env, framesToStack = 4):
        super().__init__(env, new_step_api=True)
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
import gym
import numpy as np
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
            ir += rr

        if iterminated:
            ir=-1
        return (istate, ir, iterminated, truncated, iinfo)

## Inspired by FrameStack in atari_wrappers
class StackFramesWrap(gym.Wrapper):
    def __init__(self, env, framesToStack = 4):
        super().__init__(env, new_step_api=True)
        assert framesToStack>0, "framesToStack cannot be lower tan 1 silly."
        self.env = env
        self.framesToStack = framesToStack    
        #self.frames = np.ndarray((self.framesToStack, self.observation_space))
        self.frames = [np.ndarray((1,))]*self.framesToStack
        self.observation_space = ((self.framesToStack,) + self.observation_space)

    def step(self, action):
        istate, ir, iterminated, truncated, iinfo = self.env.step(action)
        ## Note that the action is repeated, is this desirable?
        self._addFrame(istate)
        return self._getObs(), ir, iterminated, truncated, iinfo

    def reset(self):
        out, info = self.env.reset()
        #print(out.shape)
        for i in range(self.framesToStack):
            self.frames[i] = out
        return self._getObs(), info
    
    def _addFrame(self, frame):
        #print(self.frames)
        for i in range(self.framesToStack-1): 
            self.frames[i] = self.frames[i+1]
        self.frames[-1] = frame

    def _getObs(self):
        return np.stack(self.frames)

import cv2
## Assumes the frame in a width x height x 'RGB'
## proceeds to downsample the frame and grayscale it
class TransformStateWrap(gym.Wrapper):
    def __init__(self, env, downsampleMultier : int = 2):
        super().__init__(env,new_step_api=True)

        assert downsampleMultier>0, "Canot downsample to 0!"
        self.env = env
        self.downsampleMultier = downsampleMultier    

    def transform(self, frame):
        assert len(frame.shape) == 3, "Not working with an RGB image"
        cv2.imwrite("og.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        #cv2.imwrite("og.png", frame)
        print("b4", frame.max())
        newFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        print("af", newFrame.max())

        return newFrame

    def reset(self):
        out, info = self.env.reset()
        return self.transform(out), info

    def step(self, action : int):
        istate, ir, iterminated, truncated, iinfo = self.env.step(action)
        return self.transform(istate), ir, iterminated, truncated, iinfo

    def _getObs(self):
        OGFrame = self.env._getObs()
        return self.transform(OGFrame)

    
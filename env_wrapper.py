from flying_ball_env import FlyingBallGym
import utils.params_loader as pl
import utils.env_transformations as t

params = pl.load('params.yaml')
print(params.model_filename)


def getEnvBurrito():
    env = FlyingBallGym(headless=False, maxEnemies=0)
    env = t.TransformStateWrap(env, dstSize=(params.input_width, params.input_height))
    env = t.FrameSkipWrap(env, framesToSkip=params.sample_interval)
    env = t.StackFramesWrap(env, framesToStack=params.frame_stack_size)
    return env
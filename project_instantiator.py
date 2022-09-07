from rl_agent import DeepQNetwork, ConvolutionalNeuralNetwork, ReplayMemory

import env_wrapper as ew

import utils.params_loader as pl

params = pl.load('params.yaml')
print(params.model_filename)



def getProjectEnv():
    return ew.getEnvBurrito()

def getProjectModel():
    n_state = (params.frame_stack_size, params.input_width, params.input_height)
    n_action = 2
    actionsName = ["NOOP", "JUMP"]
    print(actionsName)

    device = 'cpu'
    if params.use_gpu:
        device = 'cuda'

    return DeepQNetwork(q_model=ConvolutionalNeuralNetwork(n_state[0], n_action).to(device),
                            gamma = params.gamma,
                            double_dqn=params.dual_dqn,
                            target_update_freq=params.target_update_freq,
                            learning_rate=params.learning_rate, huber=params.huber_loss,
                            clip_error=params.clip_error,
                            device=device)

def getProjectReplayMemory():
    n_state = (params.frame_stack_size, params.input_width, params.input_height)
    return ReplayMemory(n_state, memory_length=params.memory_replay_size)
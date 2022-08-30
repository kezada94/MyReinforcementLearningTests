import numpy as np
from tqdm import tqdm
import pickle
from time import sleep
from rl_agent import DeepQNetwork, ConvolutionalNeuralNetwork, ReplayMemory
from flying_ball_env import FlyingBallGym
import env_transformations as t
import torch
import typer
import mlflow

from utils.mlflow_run_decorator import mlflow_run

MODEL_FILENAME = 'lr_model.pkl'

MAX_EPISODES = 10
BATCH_SIZE = 32
MEMORY_REPLAY_SIZE = 3200
MINIMUM_FRAMES_TO_TRAIN = 32

INPUT_WIDTH = 84
INPUT_HEIGHT = 84
FRAME_STACK_SIZE = 4
SAMPLE_INTERVAL = 4

GAMMA = 0.999
LEARNING_RATE = 2e-5
HUBER_LOSS = True
DUAL_DQN = False
TARGET_UPDATE_FREQ = 100
CLIP_ERROR = True

EPSILON = 0.0

TORCH_SEED = 0

useGPU = True
def log_hyperparams():
    mlflow.log_param("MAX_EPISODES", MAX_EPISODES)
    mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
    mlflow.log_param("MEMORY_REPLAY_SIZE", MEMORY_REPLAY_SIZE)
    mlflow.log_param("MINIMUM_FRAMES_TO_TRAIN", MINIMUM_FRAMES_TO_TRAIN)
    mlflow.log_param("INPUT_WIDTH", INPUT_WIDTH)
    mlflow.log_param("INPUT_HEIGHT", INPUT_HEIGHT)
    mlflow.log_param("FRAME_STACK_SIZE", FRAME_STACK_SIZE)
    mlflow.log_param("SAMPLE_INTERVAL", SAMPLE_INTERVAL)
    mlflow.log_param("EPSILON", EPSILON)
    mlflow.log_param("TORCH_SEED", TORCH_SEED)


@mlflow_run
def train():
    torch.manual_seed(TORCH_SEED)

    # transformBurrito
    log_hyperparams()  
    env = FlyingBallGym(headless=False, maxEnemies=0)
    env = t.TransformStateWrap(env, dstSize=(INPUT_WIDTH, INPUT_HEIGHT))
    env = t.FrameSkipWrap(env, framesToSkip=SAMPLE_INTERVAL)
    env = t.StackFramesWrap(env, framesToStack=FRAME_STACK_SIZE)

    #torch.manual_seed(123)
    n_state = (FRAME_STACK_SIZE, INPUT_WIDTH, INPUT_HEIGHT)
    n_action = 2
    actionsName = ["NOOP", "JUMP"]
    print(actionsName)

    device = 'cpu'
    if useGPU:
        device = 'cuda'

    dqn_model = DeepQNetwork(q_model=ConvolutionalNeuralNetwork(n_state[0], n_action).to(device),
                            gamma = 0.999,
                            double_dqn=False,
                            target_update_freq=100,
                            learning_rate=2e-5, huber=True,
                            clip_error=True,
                            device=device)


    memory = ReplayMemory(n_state, memory_length=MEMORY_REPLAY_SIZE)

    diagnostics = {'rewards': [0], 'loss': [0],
                    'q_sum': [0], 'q_N': [0]}
    epsilon = EPSILON
    episode = 1
    terminated = False
    stacked_states, info = env.reset()


    max_reward = 0
    pbar = tqdm(total = MAX_EPISODES)
    pbar.update(1)

    while episode < MAX_EPISODES:

        state = stacked_states
        action, q = dqn_model.select_action(torch.from_numpy(state).float().unsqueeze(0).to(device),
                                    epsilon=epsilon)

        if q is not None:
            diagnostics['q_sum'][-1] += q.item()
            diagnostics['q_N'][-1] += 1

        # Aplicar la acción
        stacked_states_next, r, terminated, _, info = env.step(action) #GET NECT
        
        diagnostics['rewards'][-1] += r

        memory.push(torch.from_numpy(state).float(),
                    torch.from_numpy(stacked_states_next).float(),
                    torch.tensor(action), torch.tensor(r), terminated)

        stacked_states = stacked_states_next

        # Actualizar modelo
        if memory.pointer > MINIMUM_FRAMES_TO_TRAIN:
            mini_batch = memory.sample(BATCH_SIZE)
            if not mini_batch is None:
                diagnostics['loss'][-1] += dqn_model.update(mini_batch)

        # Preparar siguiente episodio
        if terminated:
            model = dqn_model.q_policy
            if diagnostics['rewards'][-1]>max_reward:
                max_reward = diagnostics['rewards'][-1]
                pickle.dump(model, MODEL_FILENAME)
            episode += 1
            terminated = False
            stacked_states, _ = env.reset()
            diagnostics['rewards'].append(0)
            diagnostics['loss'].append(0)
            diagnostics['q_sum'].append(0)
            diagnostics['q_N'].append(0)
            pbar.update(1)


    #for i in range(len(diagnostics['rewards'])):
    stepVals = zip(*list(diagnostics.values()))
    for i,stepVal in enumerate(stepVals):
        d={key:val for key,val in zip(diagnostics.keys(), stepVal)}
        print (d)
        mlflow.log_metrics(d, i)
    mlflow.log_metric("max_reward", max_reward)
    
    


    print(diagnostics)
    env.close()

if __name__ == "__main__":
    typer.run(train)
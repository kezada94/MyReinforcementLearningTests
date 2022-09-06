from math import inf
import numpy as np
from tqdm import tqdm
from rl_agent import DeepQNetwork, ConvolutionalNeuralNetwork, ReplayMemory
from flying_ball_env import FlyingBallGym
import env_transformations as t
import torch
import typer
import json

import utils.params_loader as pl

params = pl.load('params.yaml')
print(params.model_filename)


useGPU = True

def train():
    torch.manual_seed(params.torch_seed)

    # transformBurrito
    env = FlyingBallGym(headless=False, maxEnemies=0)
    env = t.TransformStateWrap(env, dstSize=(params.input_width, params.input_height))
    env = t.FrameSkipWrap(env, framesToSkip=params.sample_interval)
    env = t.StackFramesWrap(env, framesToStack=params.frame_stack_size)

    #torch.manual_seed(123)
    n_state = (params.frame_stack_size, params.input_width, params.input_height)
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


    memory = ReplayMemory(n_state, memory_length=params.memory_replay_size)

    diagnostics = {'rewards': [0], 'loss': [0],
                    'q_sum': [0], 'q_N': [0]}
    epsilon = params.epsilon
    episode = 0
    terminated = False
    stacked_states, info = env.reset()


    max_reward = -inf
    pbar = tqdm(total = params.max_episodes)
    while episode < params.max_episodes:

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
        if memory.pointer > params.minimum_frames_to_train:
            mini_batch = memory.sample(params.batch_size)
            if not mini_batch is None:
                diagnostics['loss'][-1] += dqn_model.update(mini_batch)

        # Preparar siguiente episodio
        if terminated:
            model = dqn_model.q_policy
            if diagnostics['rewards'][-1]>max_reward:
                max_reward = diagnostics['rewards'][-1]
                torch.save(model, params.model_filename)
            episode += 1
            terminated = False
            stacked_states, _ = env.reset()
            diagnostics['rewards'].append(0)
            diagnostics['loss'].append(0)
            diagnostics['q_sum'].append(0)
            diagnostics['q_N'].append(0)
            pbar.update(1)

    with open(params.metrics_filename, 'w', encoding='utf-8') as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=4)
    print(diagnostics)
    env.close()

if __name__ == "__main__":
    typer.run(train)
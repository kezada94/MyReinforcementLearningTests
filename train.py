from math import inf
import numpy as np
from tqdm import tqdm

import project_instantiator as pj

import torch
import typer
import json

import utils.params_loader as pl

params = pl.load('params.yaml')
print(params.model_filename)


def train():
    torch.manual_seed(params.torch_seed)
    np.random.seed(params.numpy_seed)

    env = pj.getProjectEnv()

    #n_state = (params.frame_stack_size, params.input_width, params.input_height)
    #n_action = 2
    #actionsName = ["NOOP", "JUMP"]
    #print(actionsName)

    device = 'cpu'
    if params.use_gpu:
        device = 'cuda'

    dqn_model = pj.getProjectModel()


    memory = pj.getProjectReplayMemory()

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
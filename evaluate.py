import project_instantiator as pj

import torch
import typer

import utils.params_loader as pl

params = pl.load('params.yaml')
print(params.model_filename)


def evaluate():
    torch.manual_seed(params.torch_seed)

    # transformBurrito
    env = pj.getProjectEnv()

    device = 'cpu'
    if params.use_gpu:
        device = 'cuda'

    dqn_model = pj.getProjectModel()
    dqn_model.q_policy = torch.load(params.model_filename)

    
    epsilon = params.epsilon
    terminated = False
    stacked_states, info = env.reset()

    while not terminated:

        state = stacked_states
        action, q = dqn_model.select_action(torch.from_numpy(state).float().unsqueeze(0).to(device),
                                    epsilon=epsilon)

        stacked_states_next, r, terminated, _, info = env.step(action) #GET NECT
        
        stacked_states = stacked_states_next

        
    env.close()

if __name__ == "__main__":
    typer.run(evaluate)
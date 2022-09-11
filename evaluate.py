import project_instantiator as pj

import torch
import typer

import utils.params_loader as pl
from utils.visual_artifacts_generator import plotFeatureMaps, plotFrameStack, generateConvLayersFM

params = pl.load('params.yaml')
print(params.model_filename)


def evaluate():
    torch.manual_seed(params.torch_seed)

    # transformBurrito
    env = pj.getProjectEnv()
    #env.game.addOutlierBall([45, 0], [255,255,0], 6)

    device = 'cpu'
    if params.use_gpu:
        device = 'cuda'

    dqn_model = pj.getProjectModel()
    dqn_model.q_policy = torch.load(params.model_filename)

    
    epsilon = params.epsilon
    terminated = False
    stacked_states, info = env.reset()
    #stacked_states, r, terminated, _, info = env.step(0) #GET NECT


    plotFrameStack(stacked_states)
    print(stacked_states.shape)
    stacked_states2 = torch.from_numpy(stacked_states).float().to(device)
    #stacked_states2 = torch.ones_like(stacked_states2)
    outputs = generateConvLayersFM(dqn_model.q_policy, stacked_states2)
    plotFeatureMaps(outputs[0].detach().cpu().numpy(), "Layer 0")
    plotFeatureMaps(outputs[1].detach().cpu().numpy(), "Layer 1")
    plotFeatureMaps(outputs[2].detach().cpu().numpy(), "Layer 2")
    steps = 0
    while not terminated:

        state = stacked_states
        #action, q = dqn_model.select_action(torch.from_numpy(state).float().unsqueeze(0).to(device),
        action, q = dqn_model.select_action(torch.from_numpy(state).float().unsqueeze(0).to(device),
                                    epsilon=epsilon)

        stacked_states_next, r, terminated, _, info = env.step(action) #GET NECT
        
        stacked_states = stacked_states_next

        steps += 1
    env.close()

if __name__ == "__main__":
    typer.run(evaluate)
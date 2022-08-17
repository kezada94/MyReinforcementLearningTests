import numpy as np
from tqdm import tqdm
import pickle
from time import sleep
from rl_agent import DeepQNetwork, ConvolutionalNeuralNetwork, ReplayMemory
from flying_ball_env import FlyingBallGym
import env_transformations as t
import torch

MAX_EPISODES = 10000
BATCH_SIZE = 32
MEMORY_SIZE = 3200
MINIMUM = 32

INPUT_WIDTH = 84
INPUT_HEIGHT = 84
FRAME_STACK_SIZE = 4
SAMPLE_INTERVAL = 4

useGPU = True

# transformBurrito
env = FlyingBallGym(headless=False)
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

memory = ReplayMemory(n_state, memory_length=MEMORY_SIZE)

diagnostics = {'rewards': [0], 'loss': [0],
                'q_sum': [0], 'q_N': [0]}
epsilon = 0.0
episode = 1
terminated = False
stacked_states, info = env.reset()

pbar = tqdm(total = MAX_EPISODES)
while episode < MAX_EPISODES:
#for step in tqdm(range(MAX_EPISODES)):
    #sleep(1)
    # Escoger acción
    state = stacked_states
    action, q = dqn_model.select_action(torch.from_numpy(state).float().unsqueeze(0).to(device),
                                epsilon=epsilon)


    if q is not None:
        diagnostics['q_sum'][-1] += q
        diagnostics['q_N'][-1] += 1

    # Aplicar la acción
    stacked_states_next, r, terminated, _, info = env.step(action) #GET NECT
    
    diagnostics['rewards'][-1] += r
    if terminated:
        r=-1
    # Guardar en memoria
    #env.rterminateder()

    memory.push(torch.from_numpy(state).float(),
                torch.from_numpy(stacked_states_next).float(),
                torch.tensor(action), torch.tensor(r), terminated)

    stacked_states = stacked_states_next

    # Actualizar modelo
    if memory.pointer > MINIMUM:
        mini_batch = memory.sample(BATCH_SIZE)
        if not mini_batch is None:
            diagnostics['loss'][-1] += dqn_model.update(mini_batch)

    # Preparar siguiente episodio
    if terminated:
        #print(str(np.mean(diagnostics['rewards'][-100:])))
        #if np.mean(diagnostics['rewards'][-100:])>max_reward:
        #    max_reward = np.mean(diagnostics['rewards'][-100:])
        #    print(str(max_reward) + ": modelo guardado!")
            
            #save_model(dqn_model, max_reward)
        #if episode % 10 == 0:
        #    update_plot(step, episode)
        episode += 1
        terminated = False
        stacked_states, _ = env.reset()
        diagnostics['rewards'].append(0)
        diagnostics['loss'].append(0)
        diagnostics['q_sum'].append(0)
        diagnostics['q_N'].append(0)
        pbar.update(1)
env.close()
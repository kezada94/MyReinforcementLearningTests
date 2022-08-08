import numpy as np
import gym
from tqdm import tqdm
import pickle
from time import sleep
from RLAgent import DeepQNetwork, ConvolutionalNeuralNetwork, ReplayMemory

MAX_EPISODES = 10000000
BATCH_SIZE = 32
MEMORY_SIZE = 100000
MINIMUM = 32

#torch.manual_seed(123)
n_state = (4, 84, 84)
n_action = 2
actionsName = ["NOOP", "JUMP"]
print(actionsName)

dqn_model = DeepQNetwork(q_model=ConvolutionalNeuralNetwork(n_state[0], n_action),
                        gamma = 0.999,
                        double_dqn=True,
                        target_update_freq=100,
                        learning_rate=2e-5, huber=True,
                        clip_error=True)
memory = []
if (memory is not None):
    del memory

memory = ReplayMemory(n_state, memory_length=MEMORY_SIZE)

diagnostics = {'rewards': [0], 'loss': [0],
                'q_sum': [0], 'q_N': [0]}
max_reward = 0
episode = 1
end = False
stacked_states = env.reset()
for step in tqdm(range(MAX_EPISODES)):
    # Escoger acción
    state = stacked_states
    a, q = dqn_model.select_action(torch.from_numpy(np.transpose(state.__array__(), (2,0,1))).unsqueeze(0).cuda(),
                                epsilon(episode))
    if q is not None:
        diagnostics['q_sum'][-1] += q
        diagnostics['q_N'][-1] += 1

    # Aplicar la acción
    stacked_states_next, r, end, info = env.step(a)
    
    diagnostics['rewards'][-1] += r
    if end:
        r=-1.
    # Guardar en memoria
    #env.render()
    memory.push(torch.from_numpy(np.transpose(state.__array__(), (2,0,1))).cuda(),
                torch.from_numpy(np.transpose(stacked_states_next.__array__(), (2,0,1))).cuda(),
                a, torch.tensor(r), end)

    stacked_states = stacked_states_next

    # Actualizar modelo
    if memory.pointer > MINIMUM:
        mini_batch = memory.sample(BATCH_SIZE)
        if not mini_batch is None:
            diagnostics['loss'][-1] += dqn_model.update(mini_batch)

    # Preparar siguiente episodio
    if end:
        display(str(np.mean(diagnostics['rewards'][-100:])))
        if np.mean(diagnostics['rewards'][-100:])>max_reward:
            max_reward = np.mean(diagnostics['rewards'][-100:])
            display(str(max_reward) + ": modelo guardado!")
            
            save_model(dqn_model, max_reward)
        if episode % 10 == 0:
            update_plot(step, episode)
        episode += 1
        end = False
        stacked_states = env.reset()
        diagnostics['rewards'].append(0)
        diagnostics['loss'].append(0)
        diagnostics['q_sum'].append(0)
        diagnostics['q_N'].append(0)
env.close()
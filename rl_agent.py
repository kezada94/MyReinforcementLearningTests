import matplotlib.pyplot as plt
import torch

import numpy as np
from tqdm import tqdm
import pickle
from time import sleep
import copy

class ReplayMemory:
    def __init__(self, state_dim, memory_length=1000):
        self.length = memory_length
        self.pointer = 0
        self.filled = False
        # Tensores vacíos para la historia
        self.s_current = torch.zeros((memory_length,) + state_dim)
        self.s_future = torch.zeros((memory_length,) + state_dim)
        self.a = torch.zeros(memory_length, 1, dtype=int)
        self.r = torch.zeros(memory_length, 1)
        # Adicionalmente guardaremos la condición de término
        self.end = torch.zeros(memory_length, 1, dtype=bool)

    def push(self, s_current, s_future, a, r, end):
        # Agregamos una tupla en la memoria
        self.s_current[self.pointer] = s_current
        self.s_future[self.pointer] = s_future
        self.a[self.pointer] = a
        self.r[self.pointer] = r
        self.end[self.pointer] = end
        if self.pointer + 1 == self.length:
            self.filled = True
        self.pointer =  (self.pointer + 1) % self.length

    def sample(self, size=64):
        # Extraemos una muestra aleatoria de la memoria
        if self.filled:
            idx = np.random.choice(self.length, size)
        elif self.pointer > size:
            idx = np.random.choice(self.pointer, size)
        else:
            return None
        return self.s_current[idx], self.s_future[idx], self.a[idx], self.r[idx], self.end[idx]

class ConvolutionalNeuralNetwork(torch.nn.Module):
    def __init__(self, n_input, n_output, n_filters=32, n_hidden=256):
        super(type(self), self).__init__()
        self.conv1 = torch.nn.Conv2d(n_input, n_filters, kernel_size=8, stride=4, bias=True)
        self.conv2 = torch.nn.Conv2d(n_filters, n_filters, kernel_size=4, stride=2, bias=True)
        self.conv3 = torch.nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, bias=True)
        
        #self.conv4 = torch.nn.Conv2d(n_filters, n_filters, kernel_size=2, stride=1)
        #self.conv3.weight.register_hook(lambda grad: grad * 1./np.sqrt(2))
        self.linear1 = torch.nn.Linear(7*7*32, 512)
        self.value = torch.nn.Linear(512, n_hidden)
        self.adv = torch.nn.Linear(512, n_hidden)

        self.output_adv = torch.nn.Linear(n_hidden, n_output)
        self.output_value = torch.nn.Linear(n_hidden, 1)
        
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.conv1.weight, a=2)
            torch.nn.init.kaiming_uniform_(self.conv2.weight, a=2)
            torch.nn.init.kaiming_uniform_(self.conv3.weight, a=2)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        h = self.activation(self.conv1(x))
        h = self.activation(self.conv2(h))
        h = self.activation(self.conv3(h))
        
        #h = self.activation(self.conv4(h))
        h = h.view(-1, 7*7*32)
        h = self.activation(self.linear1(h))
        v = self.activation(self.value(h))
        a = self.activation(self.adv(h))

        v = self.output_value(v)
        a = self.output_adv(a)
        out = a-a.mean()
        out = v+out
        return  out

class DeepQNetwork:
    def __init__(self, q_model, gamma=0.999, double_dqn=False, learning_rate=1e-3,
                target_update_freq=500, clip_grads=True, clip_error=False, huber=False, device='cpu'):
        self.double_dqn = double_dqn
        self.gamma = gamma
        self.q_policy = q_model
        self.n_output = q_model.output_adv.out_features
        self.clip_error = clip_error
        self.clip_grads = clip_grads
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.device = device
        if not huber:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.q_policy.parameters(), lr=learning_rate)

        if double_dqn:
            self.q_target = copy.deepcopy(self.q_policy)
            self.q_target = self.q_target
            self.q_target.eval()

    def select_action(self, state, epsilon=0.001):
        # Estrategia epsilon greedy para seleccionar acción
        if torch.rand(1).item() < 1. - epsilon:
            self.q_policy.eval()
            with torch.no_grad():
                q = self.q_policy.forward(state)[0]
                a = q.argmax().item()
                q = q[a]
            self.q_policy.train()
        else:
            q = None
            a = torch.from_numpy(np.random.randint(self.n_output, size=1))
        return a, q

    def update(self, mini_batch):
        self.update_counter += 1
        state, state_next, action, reward, end = mini_batch
        # Calcular Q
        q_current = self.q_policy(state.to(self.device)).gather(1, action.to(self.device))
        with torch.no_grad():
            if not self.double_dqn:
                q_next_best = self.q_policy(state_next.to(self.device)).max(1, keepdim=True)[0]
            else:
                action_next = self.q_policy(state_next.to(self.device)).argmax(dim=1, keepdim=True)
                q_next_best = self.q_target(state_next.to(self.device)).gather(1, action_next.to(self.device))
        # Construir el target: r + gamma*max Q(s', a')
        td_target = reward
        td_target[~end] += self.gamma*q_next_best[~end].to('cpu')
        td_target[end] = -1.
        # Calcular pérdido y sus gradientes
        self.optimizer.zero_grad()
        loss = self.criterion(q_current, td_target.to(self.device))
        if self.clip_error:
            loss.clamp_(-1., 1.)
        loss.backward()
        # Cortar gradientes grandes (mejora la estabilidad)
        if self.clip_grads:
            for param in self.q_policy.parameters():
                #param.grad.data.clamp_(-10., 10.) # usando 10 como en el paper
                param.grad.data.clamp_(-1., 1.)
            #torch.nn.utils.clip_grad.clip_grad_norm_(self.q_policy.parameters(), 10)
        # Actualizar
        self.optimizer.step()
        # Transfer policy to target
        self.transfer_policy2target()
        # Retornar el valor de la loss
        return loss.item()

    def transfer_policy2target(self):
        if self.double_dqn:
            if self.update_counter % self.target_update_freq == 0:
                self.q_target.load_state_dict(self.q_policy.state_dict())
                return True
        return False
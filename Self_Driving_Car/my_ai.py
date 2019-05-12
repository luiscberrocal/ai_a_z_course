import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class NeuralNetwork(nn.Module):

    def __init__(self, input_size, number_of_actions):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.number_of_actions = number_of_actions
        number_of_neurons = 30
        self.input_layer_to_hidden_layer_connection = nn.Linear(input_size, number_of_neurons)
        self.hidden_layer_to_output_layer_connection = nn.Linear(number_of_neurons, number_of_actions)

    def forward(self, state):
        # Activation of hidden neurons
        hidden_neurons = F.relu(self.input_layer_to_hidden_layer_connection(state))
        q_values = self.hidden_layer_to_output_layer_connection(hidden_neurons)
        return q_values


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, sample_size):
        samples = zip(*random.sample(self.memory, sample_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class DeepQNetwork(object):

    def __init__(self, input_size, number_of_actions, gamma):
        self.gamma = gamma
        self.reward_window = list()
        self.neural_network = NeuralNetwork(input_size, number_of_actions)
        self.memory = ReplayMemory(100000)
        learning_rate = 0.001
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=learning_rate)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_reward = 0
        self.last_action = 0

    def select_action(self, state):
        temperature = 7
        state_variable = Variable(state, volatile=True) * temperature
        probabilities = F.softmax(self.neural_network(state_variable))

        action = probabilities.multinomial()

        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.neural_network(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.neural_network(batch_next_state).detach().max(1)[0]
        target = self.gamma *  next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state,
                          torch.LongTensor([int(self.last_action)]),
                          torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)

        if len(self.memory.memory) > 100:
            batch_state, batch_next_state,  batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)

    def save(self):
        torch.save({'state_dict': self.neural_network.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
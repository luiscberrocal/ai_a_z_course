import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing


class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(self, number_of_actions):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.convolution_layers = list()
        self.convolution_layers[0] = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution_layers[2] = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution_layers[3] = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)

        self.fully_connected_layer = list()
        self.fully_connected_layer[0] = nn.linear()
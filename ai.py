# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#Creating NN topology
class Network(nn.Module):

    def __init__(self, input_size, actions_size):
        super(Network, self).__init__()
        self.input_size = input_size
        self.actions_size = actions_size
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, actions_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

#Experience Replay
class ReplayMem(object):
    def __init__(self, capacity):
        self.capacity = capacity;
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, sample_size):
        samples = zip(*random.sample(self.memory, sample_size))

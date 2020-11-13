#
# coding=utf-8

import torch as th
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim_obs, dim_action):
        super(Model, self).__init__()
        self.linear_a1 = nn.Linear(dim_obs, 64)
        self.linear_a2 = nn.Linear(64, 64)
        self.linear_a = nn.Linear(64, dim_action)

        self.tanh = nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        """
        :param x: tensor(batch, dim_obs)
        :return: (batch, dim_act)
        """
        x = self.LReLU(self.linear_a1(x))
        x = self.LReLU(self.linear_a2(x))
        out = self.linear_a(x)
        return out

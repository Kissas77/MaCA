#
# coding=utf-8

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_obs, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        obs_dim = dim_obs * n_agent
        act_dim = dim_action * n_agent

        self.LReLU = nn.LeakyReLU(0.01)

        self.linear_c1 = nn.Linear(obs_dim + act_dim, 128)
        self.linear_c2 = nn.Linear(128, 128)
        self.linear_c = nn.Linear(128, 1)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs, acts):
        """
        :param obs: (batch, obs_dim)
        :param acts: (batch, 4)
        :return: (batch, 1)
        """
        x_cat = self.LReLU(self.linear_c1(th.cat([obs, acts], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value


class Actor(nn.Module):
    def __init__(self, num_inputs, action_size):
        super(Actor, self).__init__()
        self.linear_a1 = nn.Linear(num_inputs, 64)
        self.linear_a2 = nn.Linear(64, 64)
        self.linear_a = nn.Linear(64, action_size)

        self.tanh = nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs, model_original_out=False):
        """
        :param obs: tensor(batch, dim_obs)
        :return: (batch, dim_act)
        """
        x = self.LReLU(self.linear_a1(obs))
        x = self.LReLU(self.linear_a2(x))
        model_out = self.linear_a(x)
        u = th.rand_like(model_out)
        policy = F.softmax(model_out - th.log(-th.log(u)), dim=-1)
        if model_original_out == True:
            return model_out, policy  # for model_out criterion
        return policy

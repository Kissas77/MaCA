#
# coding=utf-8

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from config import logger, GPU_CONFIG, MODEL_NAME, MODEL_PATH


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


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]

        self.n_agents = n_agents
        self.n_obs = dim_obs
        self.n_actions = dim_act
        self.use_cuda = GPU_CONFIG.use_cuda

        if self.use_cuda:
            logger.info("GPU Available!!")
        for i, x in enumerate(self.actors):
            if self.use_cuda:
                x.to(GPU_CONFIG.device)
                x.load_state_dict(th.load("model/{}/{}{}.pkl".format(MODEL_PATH, MODEL_NAME, i)))
            else:
                if GPU_CONFIG.use_parallel:
                    # 多gpu情况
                    x.load_state_dict({k.replace('module.', ''): v for k, v in
                                       th.load("model/{}/{}{}.pkl".format(MODEL_PATH, MODEL_NAME, i),
                                               map_location=lambda storage, loc: storage).items()})
                else:
                    x.load_state_dict(th.load("model/{}/{}{}.pkl".format(MODEL_PATH, MODEL_NAME, i),
                                              map_location=lambda storage, loc: storage))


    def select_action(self, agent_i, obs):
        """
        :param agent_i: int
        :param img_obs: ndarray
        :param info_obs: ndarray
        :return: action: ndarray
        """
        # ndarray to tensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        obs = th.unsqueeze(FloatTensor(obs), 0)
        # action
        action = self.actors[agent_i](obs).squeeze()
        logger.debug('select action: {}'.format(action))

        # tensor to ndarray
        if self.use_cuda:
            action = action.data.cpu()
        else:
            action = action.detach()
        action = action.numpy()

        return action

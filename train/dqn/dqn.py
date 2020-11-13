#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn
import numpy as np

from config import logger, GPU_CONFIG
from train.dqn.memory import ReplayMemory, Experience
from train.dqn.model import Model


# Deep Q Network off-policy
class DQN:
    def __init__(self, dim_obs, dim_act, learning_rate, reward_decay, e_greedy, replace_target_iter, batch_size,
                 e_greedy_increment, capacity):
        self.n_obs = dim_obs
        self.n_actions = dim_act

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.capacity = capacity
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.memory = ReplayMemory(capacity)

        self.use_cuda = GPU_CONFIG.use_cuda
        self.learn_step_counter = 0
        self.cost_his = []

        self.eval_net, self.target_net = Model(self.n_obs, self.n_actions), Model(self.n_obs, self.n_actions)
        if self.use_cuda:
            logger.info('GPU Available!!')
            self.eval_net = self.eval_net.cuda()
            self.target_net = self.target_net.cuda()
        self.loss_func = nn.MSELoss()
        self.optimizer = th.optim.Adam(self.eval_net.parameters(), lr=self.lr)

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')
            step_counter_str = '%09d' % self.learn_step_counter
            th.save(self.target_net.state_dict(), 'model/dqn/model_' + step_counter_str + '.pkl')

        # batch
        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))  # class(list)
        obs_batch = th.FloatTensor(np.array(batch.obs))
        logger.debug("obs_batch: {}".format(obs_batch.shape))
        action_batch = th.LongTensor(np.array(batch.action))  # (batch, 1)
        logger.debug('action batch: {}'.format(action_batch.shape))
        reward_batch = th.FloatTensor(np.array(batch.reward))  # (batch)
        logger.debug('reward_batch: {}'.format(reward_batch.shape))
        reward_batch = reward_batch.view(self.batch_size, 1)  # (batch, 1)
        logger.debug('after view reward_batch: {}'.format(reward_batch.shape))
        next_obs_batch = th.FloatTensor(np.array(batch.next_obs))
        if self.use_cuda:
            obs_batch = obs_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_obs_batch = next_obs_batch.cuda()

        # sample batch memory from all memory

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(obs_batch).gather(1, action_batch)  # shape (batch, 1)
        logger.debug(q_eval)  # todo
        q_next = self.target_net(next_obs_batch).detach()  # detach from graph, don't backpropagate
        q_target = reward_batch + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # loss store
        logger.debug('loss: {}'.format(loss.cpu().data))
        self.cost_his.append(float(loss.detach().cpu().numpy()))

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def select_action(self, obs):
        obs = th.unsqueeze(th.FloatTensor(obs), 0)  # (batch, )
        if self.use_cuda:
            obs = obs.cuda()
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net(obs)
            action = th.max(actions_value, 1)[1]
            if self.use_cuda:
                action = action.cpu()
            action = action.numpy()
        else:
            action = np.zeros(1, dtype=np.int32)
            action[0] = np.random.randint(0, self.n_actions)
        return action

#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random
from collections import namedtuple
import torch as th
import torch.nn as nn
import numpy as np

from config import GPU_CONFIG
from config import logger

# create class
Experience = namedtuple('Experience',
                        # attrs
                        ('img', 'info', 'action', 'next_img', 'next_info', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Model(nn.Module):
    def __init__(self, n_actions):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(  # 100 * 100 * 3
            nn.Conv2d(
                in_channels=5,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(  # 50 * 50 * 16
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25 * 25 * 32
        )
        self.info_fc = nn.Sequential(
            nn.Linear(3, 256),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(  # 25 * 25 * 32 + 256
            nn.Linear((25 * 25 * 32 + 256), 512),
            nn.ReLU(),
        )
        self.decision_fc = nn.Linear(512, n_actions)

    def forward(self, img, info):
        img_feature = self.conv1(img)
        img_feature = self.conv2(img_feature)
        info_feature = self.info_fc(info)
        combined = th.cat((img_feature.view(img_feature.size(0), -1), info_feature.view(info_feature.size(0), -1)),
                          dim=1)
        feature = self.feature_fc(combined)
        action = self.decision_fc(feature)
        return action


# Deep Q Network off-policy
class RLFighter:
    def __init__(
            self,
            n_actions,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=100,
            batch_size=32,
            e_greedy_increment=0.01,
            capacity=5000,
            output_graph=False,
    ):
        self.n_actions = n_actions
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

        # total learning step
        self.learn_step_counter = 0

        self.cost_his = []
        self.eval_net, self.target_net = Model(self.n_actions), Model(self.n_actions)
        if self.use_cuda:
            print('GPU Available!!')
            self.eval_net = self.eval_net.cuda()
            self.target_net = self.target_net.cuda()
        self.loss_func = nn.MSELoss()
        # self.optimizer = th.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.optimizer = th.optim.RMSprop(self.eval_net.parameters(), lr=self.lr)

    def choose_action(self, img_obs, info_obs):
        img_obs = th.unsqueeze(th.FloatTensor(img_obs), 0)
        info_obs = th.unsqueeze(th.cuda.FloatTensor(info_obs), 0)
        if self.use_cuda:
            img_obs = img_obs.cuda()
            info_obs = info_obs.cuda()
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net(img_obs, info_obs)
            action = th.max(actions_value, 1)[1]
            if self.use_cuda:
                action = action.cpu()
            action = action.numpy()
        else:
            action = np.zeros(1, dtype=np.int32)
            action[0] = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')
            step_counter_str = '%09d' % self.learn_step_counter
            th.save(self.target_net.state_dict(), 'model/simple/model_' + step_counter_str + '.pkl')
        # pre possess mem`
        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))  # class(list)
        img_batch = th.FloatTensor(np.array(batch.img))  # (batch, 5, 100, 100)
        logger.debug("img_batch: {}".format(img_batch.shape))
        info_batch = th.FloatTensor(np.array(batch.info))  # (batch, 3)
        logger.debug('info batch: {}'.format(info_batch.shape))
        action_batch = th.LongTensor(np.array(batch.action))  # (batch, 1)
        logger.debug('action batch: {}'.format(action_batch.shape))
        reward_batch = th.FloatTensor(np.array(batch.reward))  # (batch)
        logger.debug('reward_batch: {}'.format(reward_batch.shape))
        reward_batch = reward_batch.view(self.batch_size, 1)  # (batch, 1)
        logger.debug('after view reward_batch: {}'.format(reward_batch.shape))
        next_img_batch = th.FloatTensor(np.array(batch.next_img))
        next_info_batch = th.FloatTensor(np.array(batch.next_info))
        if self.use_cuda:
            img_batch = img_batch.cuda()
            info_batch = info_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_img_batch = next_img_batch.cuda()
            next_info_batch = next_info_batch.cuda()

        # sample batch memory from all memory

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(img_batch, info_batch).gather(1, action_batch)  # shape (batch, 1)
        logger.debug(q_eval)  # todo
        q_next = self.target_net(next_img_batch, next_info_batch).detach()  # detach from graph, don't backpropagate
        q_target = reward_batch + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # loss store
        logger.info('loss: {}'.format(loss.cpu().data))
        self.cost_his.append(float(loss.detach().cpu().numpy()))

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


class NetDetector(nn.Module):
    def __init__(self, n_actions):
        super(NetDetector, self).__init__()
        self.conv1 = nn.Sequential(  # 100 * 100 * 3
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(  # 50 * 50 * 16
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25 * 25 * 32
        )
        self.info_fc = nn.Sequential(
            nn.Linear(3, 256),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(  # 25 * 25 * 32 + 256
            nn.Linear((25 * 25 * 32 + 256), 512),
            nn.ReLU(),
        )
        self.decision_fc = nn.Linear(512, n_actions)

    def forward(self, img, info):
        img_feature = self.conv1(img)
        img_feature = self.conv2(img_feature)
        info_feature = self.info_fc(info)
        combined = th.cat((img_feature.view(img_feature.size(0), -1), info_feature.view(info_feature.size(0), -1)),
                          dim=1)
        feature = self.feature_fc(combined)
        action = self.decision_fc(feature)
        return action

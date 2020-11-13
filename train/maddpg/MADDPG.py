#
# coding=utf-8

from torch.optim import Adam
import torch.nn as nn
import numpy as np

import torch as th
from copy import deepcopy
from train.maddpg.memory import ReplayMemory, Experience
from config import GPU_CONFIG, logger, MODEL_PATH, IS_TEST
from train.maddpg.model_dispersed import Critic, Actor


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size, capacity, replace_target_iter,
                 episodes_before_train, learning_rate, gamma, scale_reward, is_dispersed):
        self.actors = [Actor(dim_obs, dim_act) for _ in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs, dim_act) for _ in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_obs = dim_obs
        self.n_actions = dim_act
        self.is_dispersed = is_dispersed  # 是否为离散动作
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = GPU_CONFIG.use_cuda

        self.GAMMA = gamma
        self.scale_reward = scale_reward
        self.tau = 0.01  # 替换网络比例
        self.replace_target_iter = replace_target_iter
        self.episodes_before_train = episodes_before_train
        self.learn_step_counter = 0
        self.episode_done = 0

        self.var = [1 for _ in range(n_agents)]  # 随机参数 todo 1 before 0.1
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=learning_rate) for x in self.critics]  # lr: 0.00.1
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=learning_rate) for x in self.actors]  # lr: 0.0001

        if self.use_cuda:
            logger.info("GPU Available!!")
            for i in range(len(self.actors)):
                # self.actors[i].to(GPU_CONFIG.device)
                self.actors[i] = self.actors[i].cuda()
                # 分配多个gpu
                if GPU_CONFIG.use_parallel:
                    self.actors[i] = nn.DataParallel(self.actors[i], device_ids=GPU_CONFIG.device_ids)
            for i in range(len(self.critics)):
                # self.critics[i].to(GPU_CONFIG.device)
                self.critics[i] = self.critics[i].cuda()
                if GPU_CONFIG.use_parallel:
                    self.critics[i] = nn.DataParallel(self.critics[i], device_ids=GPU_CONFIG.device_ids)
            for i in range(len(self.actors_target)):
                # self.actors_target[i].to(GPU_CONFIG.device)
                self.actors_target[i] = self.actors_target[i].cuda()
                if GPU_CONFIG.use_parallel:
                    self.actors_target[i] = nn.DataParallel(self.actors_target[i], device_ids=GPU_CONFIG.device_ids)
            for i in range(len(self.critics_target)):
                # self.critics_target[i].to(GPU_CONFIG.device)
                self.critics_target[i] = self.critics_target[i].cuda()
                if GPU_CONFIG.use_parallel:
                    self.critics_target[i] = nn.DataParallel(self.critics_target[i], device_ids=GPU_CONFIG.device_ids)

    def update_policy(self):
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))  # class(list)
            # state_batch: batch_size x n_agents x dim_obs
            obs_batch = FloatTensor(np.array(batch.obs_states))
            logger.debug('learn obs batch: {}'.format(obs_batch.shape))  # (batch, 10, dim_obs)
            action_batch = FloatTensor(np.array(batch.actions))
            logger.debug('learn action batch: {}'.format(action_batch.shape))  # torch.Size([batch, 10, 4])
            reward_batch = FloatTensor(np.array(batch.rewards))
            logger.debug('learn reward batch: {}'.format(reward_batch.shape))
            next_obs_batch = FloatTensor(np.array(batch.next_obs_states))
            logger.debug('learn next obs batch: {}'.format(next_obs_batch.shape))

            # for current agent
            whole_obs = obs_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_obs, whole_action)
            logger.debug("current Q shape: {}".format(current_Q.shape))
            logger.debug("current Q: {}".format(current_Q))

            # next actions
            next_actions = [
                self.actors_target[i](next_obs_batch[:, i, :])
                for i in range(self.n_agents)]
            next_actions = th.stack(next_actions)  # list to tensor 连接
            logger.debug("learn next action: {}".format(next_actions.shape))  # torch.Size([10, 2, 4])
            next_actions = (next_actions.transpose(0, 1).contiguous())  # todo 查查contiguous()
            logger.debug("learn next action: {}".format(next_actions.shape))  # torch.Size([2, 10, 4])

            # target q
            target_Q = self.critics_target[agent](
                next_obs_batch.view(-1, self.n_agents * self.n_obs),
                next_actions.view(-1, self.n_agents * self.n_actions)
            )
            logger.debug("target Q shape: {}".format(target_Q.shape))
            logger.debug("target Q: {}".format(target_Q))

            target_Q = (target_Q * self.GAMMA) + (
                    reward_batch[:, agent].unsqueeze(1) * self.scale_reward)
            logger.debug("reward target Q shape: {}".format(target_Q.shape))
            logger.debug("reward target Q: {}".format(target_Q))

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            # optim actor
            self.actor_optimizer[agent].zero_grad()
            obs_i = obs_batch[:, agent, :]
            if self.is_dispersed:
                model_out, action_i = self.actors[agent](obs_i, model_original_out=True)
                ac = action_batch.clone()
                ac[:, agent, :] = action_i
                whole_action = ac.view(self.batch_size, -1)
                loss_pse = th.mean(th.pow(model_out, 2))
                loss_a = th.mul(-1, th.mean(self.critics[agent](whole_obs, whole_action)))
                actor_loss = 1e-3 * loss_pse + loss_a
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 0.5)
            else:
                action_i = self.actors[agent](obs_i)
                ac = action_batch.clone()
                ac[:, agent, :] = action_i
                whole_action = ac.view(self.batch_size, -1)
                actor_loss = -self.critics[agent](whole_obs, whole_action)
                actor_loss = actor_loss.mean()
                actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        # check to replace target parameters
        if self.learn_step_counter > 0 and self.learn_step_counter % self.replace_target_iter == 0:  # todo
            logger.info('\ntarget_params_replaced\n')
            for i in range(self.n_agents):
                # replace
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

                # save model
                step_counter_str = '%09d' % self.learn_step_counter
                model_name = 'model/{}/model_{}_agent{}.pkl'.format(MODEL_PATH, step_counter_str, i)
                if not IS_TEST:
                    th.save(self.actors[i].state_dict(), model_name)

        self.learn_step_counter += 1

        logger.debug("c_loss: {}, a_loss: {}".format(len(c_loss), len(a_loss)))
        return c_loss, a_loss

    def select_action(self, agent_i, obs):
        """
        :param agent_i: int
        :param obs: ndarray
        :param is_dispersed: bool
        :return: action: ndarray
        """
        # ndarray to tensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        obs = th.unsqueeze(FloatTensor(obs), 0)  # (batch, ?)
        # action
        action = self.actors[agent_i](obs).squeeze(0)
        logger.debug('actor action: {}'.format(action))

        # 连续动作 dpg
        if not self.is_dispersed:
            # 加噪声
            action += th.from_numpy(
                np.random.randn(self.n_actions) * self.var[agent_i]).type(FloatTensor)
            if self.episode_done > self.episodes_before_train and self.var[agent_i] > 0.05:  # todo 0.05 before 0.005
                self.var[agent_i] *= 0.999998  # 噪声稀释
            logger.debug('select action+random: {}'.format(action))
            action = th.clamp(action, -1.0, 1.0)
            logger.debug('select action+clamp: {}'.format(action))

        # tensor to ndarray
        if self.use_cuda:
            action = action.data.cpu()
        else:
            action = action.detach()
        action = action.numpy()

        return action

#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import numpy as np

from agent.fix_rule.agent import Agent
from interface import Environment
from train.simple import dqn
from config import logger
from utils import reward_w, loss_w

MAP_PATH = 'maps/1000_1000_fighter10v10.map'

MAX_EPOCH = 5000
CAPACITY = 5000  # before 1e6, 500
BATCH_SIZE = 128  # not use
MAX_STEP = 500  # default 5000
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
EPSILON_INCREMENT = 0.0002
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
LEARN_INTERVAL = 100  # equal batch size

RENDER = False
DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 16
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1  # long missile attack + short missile attack + no attack
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM

if __name__ == '__main__':
    # create blue agent
    blue_agent = Agent()
    # get agent obs type
    red_agent_obs_ind = 'simple'
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    # make env
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER, max_step=MAX_STEP)
    # get map info
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    # set map info to blue agent
    blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)

    red_detector_action = []
    fighter_model = dqn.RLFighter(n_actions=ACTION_NUM, learning_rate=LR, reward_decay=GAMMA, e_greedy=EPSILON,
                                  e_greedy_increment=EPSILON_INCREMENT, capacity=CAPACITY, batch_size=BATCH_SIZE,
                                  replace_target_iter=TARGET_REPLACE_ITER)

    # execution
    for i_episode in range(MAX_EPOCH):
        step_cnt = 0
        total_reward = 0.0  # 每回合所有智能体的总体奖励
        env.reset()
        while True:
            obs_list = [0 for _ in range(red_fighter_num)]
            action_list = [0 for _ in range(red_fighter_num)]
            red_fighter_action = []
            # get obs
            if step_cnt == 0:
                red_obs_dict, blue_obs_dict = env.get_obs()  # output: raw obs结构体
            # logger.debug('blue_obs_dict: {}'.format(blue_obs_dict))

            # get action
            # get blue action
            blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, step_cnt)
            # logger.debug('blue_detector_action: {}'.format(blue_detector_action))
            # logger.debug('blue_fighter_action: {}'.format(blue_fighter_action))

            # get red action
            obs_got_ind = [False] * red_fighter_num  # 记录存活的单位
            for y in range(red_fighter_num):
                true_action = np.array([0, 1, 0, 0], dtype=np.int32)
                if red_obs_dict['fighter'][y]['alive']:
                    obs_got_ind[y] = True
                    # logger.debug("obs_got_ind: {}".format(obs_got_ind))
                    tmp_img_obs = red_obs_dict['fighter'][y]['screen']  # shape(100, 100, 5)
                    tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)  # shape(5, 100, 100)
                    logger.debug("tmp_img_obs: {}".format(tmp_img_obs.shape))
                    # logger.debug(red_obs_dict['fighter'][y].keys())
                    tmp_info_obs = red_obs_dict['fighter'][y]['info']  # shape(3, ) []
                    logger.debug('tmp_info_obs: {}'.format(tmp_info_obs.shape))
                    obs_list[y] = {'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)}

                    # dqn policy
                    tmp_action = fighter_model.choose_action(tmp_img_obs, tmp_info_obs)
                    logger.debug('tmp_action: {}'.format(tmp_action))
                    action_list[y] = tmp_action
                    # action formation
                    true_action[0] = int(360 / COURSE_NUM * int(tmp_action[0] / ATTACK_IND_NUM))
                    true_action[3] = int(tmp_action[0] % ATTACK_IND_NUM)
                    logger.info('true_action: {}'.format(true_action))

                red_fighter_action.append(true_action)
            red_fighter_action = np.array(red_fighter_action)

            # step
            env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)

            # get reward
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = env.get_reward()
            detector_reward = red_detector_reward + red_game_reward
            fighter_reward = red_fighter_reward + red_game_reward
            total_reward += fighter_reward.sum()

            # save repaly
            red_obs_dict, blue_obs_dict = env.get_obs()
            for y in range(red_fighter_num):
                if obs_got_ind[y]:
                    tmp_img_obs = red_obs_dict['fighter'][y]['screen']
                    tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                    tmp_info_obs = red_obs_dict['fighter'][y]['info']
                    fighter_model.memory.push(obs_list[y]['screen'], obs_list[y]['info'], action_list[y],
                                              copy.deepcopy(tmp_img_obs), copy.deepcopy(tmp_info_obs),
                                              fighter_reward[y])

            # if done, perform a learn
            if env.get_done():
                # detector_model.learn()
                fighter_model.learn()
                logger.info('episode: %d, reward = %f' % (i_episode, total_reward))
                logger.info('e_greedy: %f' % fighter_model.epsilon)
                # store total_reward
                reward_w(total_reward, 'train/simple/pics/reward.txt')
                break

            # if not done learn when learn interval
            if (step_cnt > 0) and (step_cnt % LEARN_INTERVAL == 0):
                # detector_model.learn()
                fighter_model.learn()

            step_cnt += 1

    # store loss
    loss_w(fighter_model.cost_his, 'train/simple/pics/loss.txt')
    logger.info('**********train finish!**************')

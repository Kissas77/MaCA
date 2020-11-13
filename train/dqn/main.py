#! /usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np

from agent.fix_rule.agent import Agent
from interface import Environment
from train.dqn.dqn import DQN
from config import logger
from train.dqn.agentutil import fighter_rule
from utils import reward_w, loss_w

RENDER = False
MAP_PATH = 'maps/1000_1000_fighter10v10.map'
AGENT_NAME = 'dqn'
# obs: 自身坐标，自身航向，主动观测列表坐标，被动观测列表频点，全局观测到的坐标
OBS_NUM = 2 + 1 + 2 * 10 + 10 + 2 * 10
DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 16
ACTION_NUM = 11

MAX_EPOCH = 5000
MAX_STEP = 500  # default 500
CAPACITY = int(1e6)  # before 1e6
BATCH_SIZE = 1  # 256
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
EPSILON_INCREMENT = 0.002
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
LEARN_INTERVAL = 100  # 100

if __name__ == '__main__':
    # create blue agent
    blue_agent = Agent()
    # get agent obs type
    red_agent_obs_ind = 'dqn'
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    # make env
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER, max_step=MAX_STEP)
    # get map info
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    # set map info to blue agent
    blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)

    red_detector_action = []
    dqn = DQN(dim_obs=OBS_NUM, dim_act=ACTION_NUM, learning_rate=LR, reward_decay=GAMMA, e_greedy=EPSILON,
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
            # todo 更改敌方的雷达频点
            for i in range(len(blue_fighter_action)):
                blue_fighter_action[i]['r_fre_point'] = i + 1

            # get red action
            obs_got_ind = [False] * red_fighter_num  # 记录存活的单位
            for y in range(red_fighter_num):
                true_action = np.array([0, 1, 0, 0], dtype=np.int32)
                if red_obs_dict['fighter'][y]['alive']:
                    obs_got_ind[y] = True
                    tmp_course = red_obs_dict['fighter'][y]['course']  # (1, )
                    tmp_pos = red_obs_dict['fighter'][y]['pos']  # (2, )
                    tmp_r_visible_pos = red_obs_dict['fighter'][y]['r_visible_pos']  # (10, 2)
                    tmp_j_visible_fp = red_obs_dict['fighter'][y]['j_visible_fp']  # (10, 1)
                    tmp_l_missile = red_obs_dict['fighter'][y]['l_missile']  # rule use
                    tmp_s_missile = red_obs_dict['fighter'][y]['s_missile']  # rule use
                    tmp_j_visible_fp = red_obs_dict['fighter'][y]['j_visible_fp']  # rule use
                    tmp_j_visible_dir = red_obs_dict['fighter'][y]['j_visible_dir']  # (10, 1)  # rule use
                    tmp_g_visible_pos = red_obs_dict['fighter'][y]['g_visible_pos']  # (10, 2)
                    # model obs change, 归一化
                    course = tmp_course / 359.
                    pos = tmp_pos / size_x
                    r_visible_pos = tmp_r_visible_pos.reshape(1, -1)[0] / size_x  # (20,)
                    j_visible_fp = tmp_j_visible_fp.reshape(1, -1)[0] / 359.  # (10,)
                    g_visible_pos = tmp_g_visible_pos.reshape(1, -1)[0] / size_x  # (20,)
                    # todo 归一化
                    obs = np.concatenate((course, pos, r_visible_pos, j_visible_fp, g_visible_pos), axis=0)
                    logger.debug('obs: {}'.format(obs))

                    obs_list[y] = obs

                    # dqn policy
                    tmp_action = dqn.select_action(obs)
                    logger.debug('tmp action: {}'.format(tmp_action))
                    # rule policy
                    true_action = fighter_rule(tmp_course, tmp_pos, tmp_l_missile, tmp_s_missile, tmp_r_visible_pos,
                                               tmp_j_visible_dir, tmp_j_visible_fp, tmp_g_visible_pos)
                    logger.debug('true aciton rule out: {}'.format(true_action))
                    # 添加动作
                    true_action[2] = tmp_action
                    logger.debug('true action: {}'.format(true_action))

                    action_list[y] = tmp_action
                red_fighter_action.append(true_action)

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
                    tmp_course = red_obs_dict['fighter'][y]['course']  # (1, )
                    tmp_pos = red_obs_dict['fighter'][y]['pos']  # (2, )
                    tmp_r_visible_pos = red_obs_dict['fighter'][y]['r_visible_pos']  # (10, 2)
                    tmp_j_visible_fp = red_obs_dict['fighter'][y]['j_visible_fp']  # (10, 1)
                    tmp_g_visible_pos = red_obs_dict['fighter'][y]['g_visible_pos']  # (10, 2)
                    # model obs change, 归一化
                    course = tmp_course / 359.
                    pos = tmp_pos / size_x
                    r_visible_pos = tmp_r_visible_pos.reshape(1, -1)[0] / size_x  # (20,)
                    j_visible_fp = tmp_j_visible_fp.reshape(1, -1)[0] / 359.  # (10,)
                    g_visible_pos = tmp_g_visible_pos.reshape(1, -1)[0] / size_x  # (20,)
                    # todo 归一化
                    obs = np.concatenate((course, pos, r_visible_pos, j_visible_fp, g_visible_pos), axis=0)

                    # store
                    dqn.memory.push(obs_list[y], action_list[y], copy.deepcopy(obs), fighter_reward[y])

            # if done, perform a learn
            if env.get_done():
                # detector_model.learn()
                dqn.learn()
                logger.info('episode: %d, reward = %f' % (i_episode, total_reward))
                logger.info('e_greedy: %f' % dqn.epsilon)
                # store total_reward
                reward_w(total_reward, 'train/dqn/pics/reward.txt')
                break

            # if not done learn when learn interval
            if (step_cnt > 0) and (step_cnt % LEARN_INTERVAL == 0):
                dqn.learn()

            step_cnt += 1
    # store loss
    loss_w(dqn.cost_his, 'train/dqn/pics/loss.txt')
    logger.info('**********train finish!**************')

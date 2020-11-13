#
# coding=utf-8

"""
    maddpg：
    1 obs：
    2 action： 偏角：连续/离散动作
"""

import numpy as np
import torch as th

from train.maddpg.MADDPG import MADDPG
from interface import Environment
from agent.fix_rule.agent import Agent  # todo 更改规则模型
from config import logger, IS_DISPERSED, PICS_PATH, IS_TEST
from rule.agentutil_stable import fighter_rule
from utils import reward_w, loss_w, action2direction

MAX_EPOCH = 10000  # 2000
MAX_STEP = 800  # default 5000 after: 800
STEP_BEFORE_TRAIN = 0  # 前多少步不需要 100
BATCH_SIZE = 256  # 256
TARGET_REPLACE_ITER = 50  # target update frequency 50
CAPACITY = int(1e6)  # before 1e6, 500
LEARN_INTERVAL = 100  # 学习间隔 100
EPISODES_BEFORE_TRAIN = 50  # 开始训练前的回合数 50
LR = 0.01  # learning rate
GAMMA = 0.999  # reward discount
SCALE_REWARD = 1  # 奖励缩放  # 0.01

RENDER = False
MAP_PATH = 'maps/1000_1000_fighter10v10.map'
AGENT_NAME = 'maddpg'
# # obs: 自身坐标，自身航向，短导弹剩余，长导弹剩余，主动观测列表坐标，被动观测航向，全局观测坐标
# OBS_NUM = 2 + 1 + 1 + 1 + 2 * 10 + 10 + 2 * 10
# obs: 自身坐标，自身航向，短导弹剩余，长导弹剩余，主动观测列表坐标，被动观测航向，打击id
OBS_NUM = 2 + 1 + 1 + 1 + 2 * 10 + 10 + 10  # todo before 全局观测坐标
# action
DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 16
ACTION_NUM = 21 if IS_DISPERSED else 1  # dispersed num or ddpg

if __name__ == '__main__':
    # reward_record = []  # 记录每轮训练的奖励
    agent0_c_loss = []
    agent0_a_loss = []
    # get agent obs type
    blue_agent = Agent()  # blue agent
    red_agent_obs_ind = AGENT_NAME  # red agent
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    # init model
    maddpg = MADDPG(n_agents=FIGHTER_NUM, dim_obs=OBS_NUM, dim_act=ACTION_NUM,
                    batch_size=BATCH_SIZE, capacity=CAPACITY, replace_target_iter=TARGET_REPLACE_ITER,
                    episodes_before_train=EPISODES_BEFORE_TRAIN, learning_rate=LR, gamma=GAMMA,
                    scale_reward=SCALE_REWARD, is_dispersed=IS_DISPERSED)
    # gpu
    FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
    # make env
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER, max_step=MAX_STEP,
                      random_pos=True)
    # get map info
    size_x, size_y = env.get_map_size()  # size_x == size_y == 1000
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    # set map info to blue agent
    blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)
    red_detector_action = []  # temp

    for i_episode in range(MAX_EPOCH):
        step_cnt = 0
        env.reset()
        total_reward = 0.0  # 每回合所有智能体的总体奖励
        rr = np.zeros((FIGHTER_NUM,))  # 每回合每个智能体的奖励

        # get obs
        red_obs_dict, blue_obs_dict = env.get_obs()  # output: raw obs结构体

        while True:
            # obs_list = []
            obs_list = []  # len == n agents
            action_list = []  # # len == n agents
            red_fighter_action = []  # # len == n agents

            # get blue action
            blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, step_cnt)

            # get red action
            for y in range(red_fighter_num):
                tmp_course = red_obs_dict['fighter'][y]['course']  # (1, )
                tmp_pos = red_obs_dict['fighter'][y]['pos']  # (2, )
                tmp_l_missile = red_obs_dict['fighter'][y]['l_missile']  # (1, )
                tmp_s_missile = red_obs_dict['fighter'][y]['s_missile']  # (1, )
                tmp_r_visible_pos = red_obs_dict['fighter'][y]['r_visible_pos']  # (10, 2)
                tmp_j_visible_fp = red_obs_dict['fighter'][y]['j_visible_fp']  # rule use
                tmp_j_visible_dir = red_obs_dict['fighter'][y]['j_visible_dir']  # (10, 1)
                tmp_g_striking_pos = red_obs_dict['fighter'][y]['g_striking_pos']  # (10, 2)
                tmp_r_visible_dis = red_obs_dict['fighter'][y]['r_visible_dis']  # (10, 1)
                tmp_striking_id = red_obs_dict['fighter'][y]['striking_id']
                # model obs change, 归一化
                if step_cnt > STEP_BEFORE_TRAIN:
                    course = tmp_course / 359.
                    pos = tmp_pos / size_x
                    l_missile = tmp_l_missile / 2.
                    s_missile = tmp_s_missile / 4.
                    r_visible_pos = tmp_r_visible_pos.reshape(1, -1)[0] / size_x  # (20,)
                    j_visible_dir = tmp_j_visible_dir.reshape(1, -1)[0] / 359  # (10,)
                    # g_striking_pos = tmp_g_striking_pos.reshape(1, -1)[0] / size_x  # (20, )  # todo
                    striking_id = tmp_striking_id.reshape(1, -1)[0] / 1
                    obs = np.concatenate(
                        (course, pos, l_missile, s_missile, r_visible_pos, j_visible_dir, striking_id), axis=0)
                    logger.debug('obs: {}'.format(obs))
                    obs_list.append(obs)

                # true action
                true_action = np.array([0, 1, 0, 0], dtype=np.int32)
                if not red_obs_dict['fighter'][y]['alive']:
                    # 如果有智能体已经死亡，则默认死亡动作输出
                    action_list.append(np.array([-1 for _ in range(ACTION_NUM)], dtype=np.float32))
                else:
                    # rule policy
                    true_action = fighter_rule(tmp_course, tmp_pos, tmp_l_missile, tmp_s_missile, tmp_r_visible_pos,
                                               tmp_r_visible_dis, tmp_j_visible_dir, tmp_j_visible_fp,
                                               tmp_striking_id, tmp_g_striking_pos, step_cnt)
                    true_action[2] = 11  # todo 更改干扰
                    logger.debug('true action rule out: {}'.format(true_action))
                    # 添加动作, 将动作转换为偏角
                    if step_cnt > STEP_BEFORE_TRAIN:
                        # model policy
                        # if any([any(r_visible_pos >= 0), any(j_visible_dir >= 0)]):
                        #     tmp_action = np.array([0. for _ in range(ACTION_NUM)], dtype=np.float32)
                        #     tmp_action[0] = 1.
                        # else:
                        tmp_action = maddpg.select_action(y, obs)
                        tmp_action_i = np.random.choice(tmp_action.shape[0], p=tmp_action.ravel())  # 根据概率选动作
                        # tmp_action_i = np.argmax(tmp_action)
                        logger.debug('tmp action i: {}'.format(tmp_action_i))
                        true_action[0] = action2direction(true_action[0], tmp_action_i, ACTION_NUM)

                        logger.debug('tmp action: {}'.format(tmp_action))
                        action_list.append(tmp_action)
                logger.debug('true action: {}'.format(true_action))
                red_fighter_action.append(true_action)

            # env step
            logger.info('agent0 true action: {}'.format(red_fighter_action[0]))  # test
            red_fighter_action = np.array(red_fighter_action)
            env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)

            # get reward
            if step_cnt > STEP_BEFORE_TRAIN:
                red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = env.get_reward()
                detector_reward = red_detector_reward + red_game_reward
                fighter_reward = red_fighter_reward + red_game_reward
                total_reward += fighter_reward.sum()
                rr += fighter_reward

            # get next obs
            red_obs_dict, blue_obs_dict = env.get_obs()

            # store replay
            # next obs
            if step_cnt > STEP_BEFORE_TRAIN:
                next_obs_list = []
                for y in range(red_fighter_num):
                    tmp_course = red_obs_dict['fighter'][y]['course']  # (1, )
                    tmp_pos = red_obs_dict['fighter'][y]['pos']  # (2, )
                    tmp_l_missile = red_obs_dict['fighter'][y]['l_missile']  # (1, )
                    tmp_s_missile = red_obs_dict['fighter'][y]['s_missile']  # (1, )
                    tmp_r_visible_pos = red_obs_dict['fighter'][y]['r_visible_pos']  # (10, 2)
                    tmp_j_visible_dir = red_obs_dict['fighter'][y]['j_visible_dir']  # (10, 1)
                    tmp_g_visible_pos = red_obs_dict['fighter'][y]['g_striking_pos']  # (10, 2)
                    tmp_striking_id = red_obs_dict['fighter'][y]['striking_id']
                    # model obs change, 归一化
                    course = tmp_course / 359.
                    pos = tmp_pos / size_x
                    l_missile = tmp_l_missile / 2.
                    s_missile = tmp_s_missile / 4.
                    r_visible_pos = tmp_r_visible_pos.reshape(1, -1)[0] / size_x  # (20,)
                    j_visible_dir = tmp_j_visible_dir.reshape(1, -1)[0] / 359  # (10,)
                    # g_visible_pos = tmp_g_visible_pos.reshape(1, -1)[0] / size_x  # (20,)
                    striking_id = tmp_striking_id.reshape(1, -1)[0] / 1
                    obs = np.concatenate(
                        (course, pos, l_missile, s_missile, r_visible_pos, j_visible_dir, striking_id),
                        axis=0)
                    next_obs_list.append(obs)

            # store
            if step_cnt > STEP_BEFORE_TRAIN:
                maddpg.memory.push(obs_list, action_list, next_obs_list, fighter_reward)

            # if done, perform a learn
            if env.get_done():
                if maddpg.episode_done > maddpg.episodes_before_train:
                    logger.info('done and training now begins...')
                    c_loss, a_loss = maddpg.update_policy()
                    agent0_c_loss = float(c_loss[0].data.cpu().numpy())
                    agent0_a_loss = float(a_loss[0].data.cpu().numpy())
                    # save loss
                    if not IS_TEST:
                        loss_w(agent0_c_loss, 'train/{}/pics/agent0_c_loss.txt'.format(PICS_PATH))
                        loss_w(agent0_a_loss, 'train/{}/pics/agent0_a_loss.txt'.format(PICS_PATH))
                break
            # if not done learn when learn interval
            if maddpg.episode_done > maddpg.episodes_before_train and (step_cnt % LEARN_INTERVAL == 0):
                logger.info('training now begins...')
                c_loss, a_loss = maddpg.update_policy()
                agent0_c_loss = float(c_loss[0].data.cpu().numpy())
                agent0_a_loss = float(a_loss[0].data.cpu().numpy())
                # save loss
                if not IS_TEST:
                    loss_w(agent0_c_loss, 'train/{}/pics/agent0_c_loss.txt'.format(PICS_PATH))
                    loss_w(agent0_a_loss, 'train/{}/pics/agent0_a_loss.txt'.format(PICS_PATH))

            step_cnt += 1
            logger.info("Episode: {}, step: {}".format(maddpg.episode_done, step_cnt))

        maddpg.episode_done += 1
        logger.info('Episode: %d, reward = %f' % (i_episode, total_reward))
        # store total_reward
        if not IS_TEST:
            reward_w(total_reward, 'train/{}/pics/reward.txt'.format(PICS_PATH))

    logger.info('**********train finish!**************')

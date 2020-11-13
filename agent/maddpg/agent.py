#
# coding=utf-8


import os
import numpy as np

from agent.base_agent import BaseAgent
from config import logger, IS_DISPERSED, MODEL_NAME, MODEL_PATH
from utils import action2direction
from rule.agentutil_stable import fighter_rule

if IS_DISPERSED:
    from agent.maddpg.maddpg_dispersed import MADDPG
else:
    from agent.maddpg.maddpg import MADDPG

DETECTOR_NUM = 0
FIGHTER_NUM = 10
OBS_NUM = 2 + 1 + 2 * 10 + 10 + 10 + 2 * 10
ACTION_NUM = 21 if IS_DISPERSED else 1  # # dispersed num, ddpg
STEP_BEFORE_TRAIN = 0


class Agent(BaseAgent):
    def __init__(self):
        """
        Init this agent
        :param size_x: battlefield horizontal size
        :param size_y: battlefield vertical size
        :param detector_num: detector quantity of this side
        :param fighter_num: fighter quantity of this side
        """
        BaseAgent.__init__(self)
        self.obs_ind = 'maddpg'
        if not os.path.exists('model/{}/{}0.pkl'.format(MODEL_PATH, MODEL_NAME)):
            logger.info('Error: agent maddpg model data not exist!')
            exit(1)
        self.maddpg = MADDPG(FIGHTER_NUM, OBS_NUM, ACTION_NUM)

    def set_map_info(self, size_x, size_y, detector_num, fighter_num):
        self.size_x = size_x
        self.size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num

    def __reset(self):
        pass

    def get_action(self, obs_dict, step_cnt):
        """
        get actions
        :param detector_obs_list:
        :param fighter_obs_list:
        :param joint_obs_dict:
        :param step_cnt:
        :return:
        """
        detector_action = []
        fighter_action = []
        for y in range(self.fighter_num):
            tmp_course = obs_dict['fighter'][y]['course']  # (1, )
            tmp_pos = obs_dict['fighter'][y]['pos']  # (2, )
            tmp_r_visible_pos = obs_dict['fighter'][y]['r_visible_pos']  # (10, 2)
            tmp_r_visible_dis = obs_dict['fighter'][y]['r_visible_dis']  # (10, 1)
            tmp_l_missile = obs_dict['fighter'][y]['l_missile']  # rule use
            tmp_s_missile = obs_dict['fighter'][y]['s_missile']  # rule use
            tmp_j_visible_fp = obs_dict['fighter'][y]['j_visible_fp']  # rule use
            tmp_j_visible_dir = obs_dict['fighter'][y]['j_visible_dir']  # (10, 1)
            tmp_g_striking_pos = obs_dict['fighter'][y]['g_striking_pos']  # (10, 2)
            tmp_striking_id = obs_dict['fighter'][y]['striking_id']  # (10, 1)
            # model obs change, 归一化
            if step_cnt > STEP_BEFORE_TRAIN:
                course = tmp_course / 359.
                pos = tmp_pos / self.size_x
                r_visible_pos = tmp_r_visible_pos.reshape(1, -1)[0] / self.size_x  # (20,)
                j_visible_dir = tmp_j_visible_dir.reshape(1, -1)[0] / 359  # (10,)
                striking_id = tmp_striking_id.reshape(1, -1)[0] / 1
                # g_striking_pos = tmp_g_striking_pos.reshape(1, -1)[0] / self.size_x  # (20,)
                g_striking_pos = np.full((20,), -0.001)
                obs = np.concatenate((course, pos, r_visible_pos, j_visible_dir, striking_id, g_striking_pos), axis=0)
                logger.debug('obs: {}'.format(obs))

            true_action = np.array([0, 1, 0, 0], dtype=np.int32)
            if obs_dict['fighter'][y]['alive']:
                # rule policy
                true_action = fighter_rule(tmp_course, tmp_pos, tmp_l_missile, tmp_s_missile, tmp_r_visible_pos,
                                           tmp_r_visible_dis, tmp_j_visible_dir, tmp_j_visible_fp, tmp_striking_id,
                                           tmp_g_striking_pos, step_cnt)
                logger.debug('true action rule out: {}'.format(true_action))
                # model policy
                if step_cnt > STEP_BEFORE_TRAIN and not any(
                        [any(r_visible_pos >= 0), any(j_visible_dir >= 0)]):
                    tmp_action = self.maddpg.select_action(y, obs)
                    logger.debug('tmp action: {}'.format(tmp_action))
                    # 添加动作, 将动作转换为偏角
                    tmp_action_i = np.argmax(tmp_action)
                    logger.debug('tmp action i: {}'.format(tmp_action_i))
                    true_action[0] = action2direction(true_action[0], tmp_action_i, ACTION_NUM)
                    logger.debug('course: {}'.format(true_action[0]))

            logger.debug('true action: {}'.format(true_action))
            fighter_action.append(true_action)
        fighter_action = np.array(fighter_action)

        return detector_action, fighter_action

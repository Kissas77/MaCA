#
# coding=utf-8

"""
    为训练添加规则
"""

import math
import numpy as np

from config import logger
import random


def getarc360(arc):
    """
    航向转换
    :param arc: int, 航向
    :return:
    """
    if 0 <= arc <= 180:
        return arc
    elif -180 <= arc < 0:
        return 360 + arc


def fighter_rule(tmp_course, tmp_pos, tmp_l_missile, tmp_s_missile, tmp_r_visible_pos, tmp_j_visible_dir,
                 tmp_j_visible_fp, tmp_g_visible_pos):
    """
    飞机打击规则
    1 根据雷达观测>被动观测>全局观测获取地方单位，获取到敌方单位id或者频点
    2 根据远、近导弹剩余量判断使用什么武器打击
    3 进行打击或者干扰
    4 如果没有发现敌人，根据自身位置航行
    5 如果没有导弹，将雷达频点与干扰频点关闭
    :param tmp_j_visible_fp: dict (10,1)            #todo：干扰
    :param tmp_course: dict (1,1)
    :param tmp_pos: dict (1,2)
    :param tmp_l_missile: dict (1,1)
    :param tmp_s_missile: dict (1,1)
    :param tmp_r_visible_pos: dict (10,2)
    :param tmp_j_visible_dir: dict (10,1)
    :param tmp_g_visible_pos: dict (10,2)
    :return:  true_action: array, 战斗机行动
    """

    tmp_r_visible_pos = tmp_r_visible_pos.transpose(1, 0)  # (2,10)
    tmp_j_visible_dir = tmp_j_visible_dir.transpose(1, 0)  # (1,10)
    tmp_j_visible_fp = tmp_j_visible_fp.transpose(1, 0)  # (1,10)      # todo:干扰
    tmp_g_visible_pos = tmp_g_visible_pos.transpose(1, 0)  # (2,10)
    # 判断是否战斗单元侦测到可攻击的对象
    farr = np.where(tmp_r_visible_pos[0] > 0)  # 主动观测列表 farr = [id, id'] # todo 是一个飞机一个点么
    farr1 = np.where(tmp_j_visible_dir[0] > 0)  # 全局观测列表
    farr2 = np.where(tmp_g_visible_pos[0] > 0)  # 被动观测列表

    true_action = np.array([0, 1, 1, 0], dtype=np.int32)

    if tmp_l_missile[0] == 0 and tmp_s_missile[0] > 0:
        # 雷达观测到有敌人
        if len(farr[0]) > 0:
            logger.debug('雷达观测到敌人............')
            id_ = random.choice(farr[0])
            fightpos = [tmp_r_visible_pos[0][id_], tmp_r_visible_pos[1][id_]]
            action_id = id_ + 1  # id为索引号+1
            true_action[0] = getarc360(
                int(math.atan2(fightpos[1] - tmp_pos[1], fightpos[0] - tmp_pos[0]) * 180 / math.pi))
            true_action[3] = action_id + 10
        # 战机被动观测列表观测到敌人
        elif len(farr2[0]) > 0:
            logger.debug('len(farr2[0]: {}'.format(len(farr2[0])))
            logger.debug('被动观测列表敌人频点............')
            id_ = random.choice(farr2[0])
            action_id = id_ + 1
            true_action[2] = tmp_j_visible_fp[0][id_]  # todo:干扰
            true_action[0] = tmp_j_visible_dir[0][id_]
            true_action[3] = action_id + 10  # 在观测列表中选第一个目标时为id最小目标（与之前优先选择先观测到的目标不同）

        # joint全局列表被动观测到有敌人
        elif len(farr1[0]) > 0:
            logger.debug('joint列表被动观测到有敌人............')
            id_ = random.choice(farr1[0])
            fightpos = [tmp_g_visible_pos[0][id_], tmp_g_visible_pos[1][id_]]
            action_id = id_ + 1
            true_action[0] = getarc360(
                int(math.atan2(fightpos[1] - tmp_pos[1], fightpos[0] - tmp_pos[0]) * 180 / math.pi))
            true_action[3] = action_id + 10
        else:
            logger.debug('没有探测到敌人............')
            if tmp_pos[0] == 1000:
                true_action[0] = 180
            elif tmp_pos[0] == 10:
                true_action[0] = 0
            else:
                true_action[0] = tmp_course[0]
            if tmp_pos[1] == 0:
                true_action[0] = 90
            if tmp_pos[1] == 1000:
                true_action[0] = 270

    elif tmp_l_missile[0] > 0:
        if len(farr[0]) > 0:
            # logger.wait()
            id_ = random.choice(farr[0])
            true_action[3] = id_ + 1
            logger.debug('雷达发现敌人长导弹打击: {}'.format(true_action[3]))
        elif len(farr2[0]) > 0:
            id_ = random.choice(farr2[0])
            true_action[3] = id_ + 1
            true_action[2] = tmp_j_visible_fp[0][id_]  # todo: 干扰
            logger.debug('被动观测发现敌人长导弹打击: {}'.format(true_action[3]))
        elif len(farr1[0]) > 0:
            id_ = random.choice(farr1[0])
            true_action[3] = id_ + 1
            logger.debug('全局被动发现敌人长导弹打击: {}'.format(true_action[3]))

        else:
            logger.debug('.............没有探测到敌人............')
            if tmp_pos[0] == 1000:
                true_action[0] = 180
            elif tmp_pos[0] == 10:
                true_action[0] = 0
            else:
                true_action[0] = tmp_course[0]
            if tmp_pos[1] == 0:
                true_action[0] = 90
            if tmp_pos[1] == 1000:
                true_action[0] = 270

    elif tmp_l_missile[0] == 0 and tmp_s_missile[0] == 0:
        true_action[1] = 0
        if len(farr1[0]) > 0:
            id_ = random.choice(farr1[0])
            fightpos = [tmp_g_visible_pos[0][id_], tmp_g_visible_pos[1][id_]]
            e_course = getarc360(
                int(math.atan2(fightpos[1] - tmp_pos[1], fightpos[0] - tmp_pos[0]) * 180 / math.pi))
            true_action[0] = 360 - e_course
        elif len(farr2[0]) > 0:
            id_ = random.choice(farr2[0])
            true_action[0] = 360 - tmp_j_visible_dir[0][id_]
            # true_action[2] = tmp_j_visible_fp[0][id_]
        else:
            if tmp_pos[0] == 1000 or tmp_pos[0] == 0:
                if tmp_pos[1] == 1000:
                    true_action[0] = 180
                elif tmp_pos[1] == 0:
                    true_action[0] = 0
                elif tmp_pos[1] > 500:
                    true_action[0] = 90
                else:
                    true_action[0] = 270
            elif tmp_pos[1] == 0 or tmp_pos[1] == 1000:
                if tmp_pos[0] < 500 and tmp_pos[0] != 0:
                    true_action[0] = 0
                elif tmp_pos[0] >= 500 and tmp_pos[0] != 1000:
                    true_action[0] = 180

    return true_action

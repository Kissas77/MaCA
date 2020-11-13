#
# coding=utf-8

"""
最新规则2：根据目标距离选长远导弹打击，提高命中率
        选择距离最近的为打击目标
        干扰点根据被动观测敌人频点变化与否设置11或对应频点
"""

import math
import numpy as np
from config import logger
import random

g_j_fp_list = np.full((10, 1), -1, dtype=np.int32)


def judge_action_id(tmp_striking_id):

    while True:
        true_action3 = random.randint(1, 20)
        if true_action3 > 10:
            id = true_action3 - 10
        else:
            id = true_action3
        if tmp_striking_id[id - 1] == 0:
            break

    return true_action3


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


def fighter_rule(tmp_course, tmp_pos, tmp_l_missile, tmp_s_missile, tmp_r_visible_pos, tmp_r_visible_dis,
                 tmp_j_visible_dir, tmp_j_visible_fp, tmp_striking_id, tmp_g_striking_pos, step_cnt):
    """
    飞机打击规则
    1 根据雷达观测>被动观测>全局观测获取地方单位，获取到敌方单位id或者频点
    2 根据远、近导弹剩余量判断使用什么武器打击
    3 进行打击或者干扰
    4 如果没有发现敌人，根据自身位置航行
    5 如果没有导弹，将雷达频点与干扰频点关闭
    :param tmp_striking_id:
    :param tmp_r_visible_dis:
    :param tmp_j_visible_fp: dict (10,1)            #  todo：干扰
    :param tmp_course: dict (1,1)
    :param tmp_pos: dict (1,2)
    :param tmp_l_missile: dict (1,1)
    :param tmp_s_missile: dict (1,1)
    :param tmp_r_visible_pos: dict (10,2)
    :param tmp_j_visible_dir: dict (10,1)
    :param tmp_g_striking_pos: dict (10,2)
    :param step_cnt: int
    :return:  true_action: array, 战斗机行动
    """
    global g_j_fp_list
    if step_cnt == 1:
        g_j_fp_list = np.full((10, 1), -1, dtype=np.int32)

    tmp_r_visible_pos = tmp_r_visible_pos.transpose(1, 0)  # (2,10)
    tmp_j_visible_dir = tmp_j_visible_dir.transpose(1, 0)  # (1,10)
    tmp_j_visible_fp = tmp_j_visible_fp.transpose(1, 0)  # (1,10)  # todo:干扰
    tmp_striking_id = tmp_striking_id.transpose(1, 0)[0]  # (1,10)
    tmp_g_striking_pos = tmp_g_striking_pos.transpose(1, 0)  # (2, 10)
    tmp_r_visible_dis = tmp_r_visible_dis.transpose(1, 0)[0]  # (1,10)

    # 判断是否战斗单元侦测到可攻击的对象
    # 判断是否战斗单元侦测到可攻击的对象
    farr = np.where(tmp_r_visible_pos[0] >= 0)  # 主动观测列表 farr = [id, id'] # todo 是一个飞机一个点么
    farr2 = np.where(tmp_j_visible_dir[0] >= 0)  # 被动观测列表

    true_action = np.array([0, 1, 11, 0], dtype=np.int32)  # todo 初始干扰
    true_action[3] = judge_action_id(tmp_striking_id)  # 初始打击目标随机
    true_action[0] = tmp_course  # 固定航向
    true_action[1] = random.randint(1, 10)  # 雷达频点随机

    # 雷达观测到有敌人
    if len(farr[0]) > 0:
        logger.debug('雷达观测到敌人............')
        id_ = np.argmin(tmp_r_visible_dis)  # todo 是否随机
        fight_pos = [tmp_r_visible_pos[0][id_], tmp_r_visible_pos[1][id_]]
        action_id = id_ + 1  # id为索引号+1
        true_action[0] = getarc360(
            int(math.atan2(fight_pos[1] - tmp_pos[1], fight_pos[0] - tmp_pos[0]) * 180 / math.pi))
        if tmp_striking_id[id_] == 0:
            if tmp_s_missile[0] > 0 and tmp_r_visible_dis[id_] <= 50:
                true_action[3] = action_id + 10
            elif tmp_l_missile[0] > 0:
                true_action[3] = action_id
            elif tmp_l_missile[0] == 0 and tmp_s_missile[0] == 0:
                true_action[0] = 180 + getarc360(
                    int(math.atan2(fight_pos[1] - tmp_pos[1], fight_pos[0] - tmp_pos[0]) * 180 / math.pi))

    # 战机被动观测列表观测到敌人
    elif len(farr2[0]) > 0:
        logger.debug('被动观测列表敌人频点............')
        id_ = random.choice(farr2[0])
        action_id = id_ + 1
        true_action[0] = tmp_j_visible_dir[0][id_]
        if tmp_striking_id[id_] == 0:
            if tmp_l_missile[0] > 0:
                true_action[3] = action_id
            elif tmp_s_missile[0] > 0:
                true_action[3] = action_id + 10
            elif tmp_l_missile[0] == 0 and tmp_s_missile[0] == 0:
                true_action[0] = getarc360(180 - tmp_j_visible_dir[0][id_])

    else:
        logger.debug('没有探测到敌人............')
        if tmp_l_missile[0] > 0 or tmp_s_missile[0] > 0:
            if tmp_pos[0] == 900:
                true_action[0] = 180
            elif tmp_pos[0] == 10:
                true_action[0] = 0
            else:
                true_action[0] = tmp_course[0]
            if tmp_pos[1] == 10:
                true_action[0] = 90
            if tmp_pos[1] == 900:
                true_action[0] = 270
        else:
            if tmp_pos[0] == 0:
                if tmp_pos[1] == 0:
                    true_action[0] = 90 if 45 < tmp_course < 225 else 0
                elif tmp_pos[1] == 1000:
                    true_action[0] = 270 if 135 < tmp_course < 315 else 0
                else:
                    true_action[0] = 270 if tmp_course > 180 else 90
            elif tmp_pos[0] == 1000:
                if tmp_pos[1] == 0:
                    true_action[0] = 180 if 135 < tmp_course < 315 else 90
                elif tmp_pos[1] == 1000:
                    true_action[0] = 180 if 45 < tmp_course < 225 else 270
                else:
                    true_action[0] = 270 if tmp_course > 180 else 90
            elif tmp_pos[1] == 0 and tmp_pos[0] != 0 and tmp_pos[0] != 1000:
                true_action[0] = 0 if tmp_course > 270 else 180
            elif tmp_pos[1] == 1000 and tmp_pos[0] != 0 and tmp_pos[0] != 1000:
                true_action[0] = 0 if tmp_course < 90 else 180

    # 干扰频点处理
    if all(g_j_fp_list < 0) and len(farr2[0] > 0):
        for i in farr2[0]:
            g_j_fp_list[i] = tmp_j_visible_fp[0][i]
    elif len(farr2[0] > 0):
        for i in farr2[0]:
            if tmp_j_visible_fp[0][i] == g_j_fp_list[i]:
                true_action[2] = tmp_j_visible_fp[0][i]
            else:
                true_action[2] = 11

    if step_cnt == 1:
        # 第一步的时候设置航向
        arc_center = getarc360(
            int(math.atan2(500 - tmp_pos[1], 500 - tmp_pos[0]) * 180 / math.pi))
        if 90 < arc_center < 270:
            true_action[0] = 180
        else:
            true_action[0] = 0

    return true_action

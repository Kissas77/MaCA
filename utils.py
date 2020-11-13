#
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch as th
import argparse

from config import PICS_PATH


def gen_acitons(x, y, n, m):
    """
    :param x: int
    :param y: int
    :param n: int
    :param m: int
    :return: list
    """
    action_list = []
    for x_i in range(x):
        for y_i in range(y):
            for n_i in range(n):
                for m_i in range(m):
                    action = [0 for _ in range(x + y + n + m)]
                    action[x_i] = 1
                    action[y_i + x] = 1
                    action[n_i + x + y] = 1
                    action[m_i + x + y + n] = 1
                    action_list.append(action)
    return action_list


def gen_actions_by_bin(x, y, n, m):
    """
    :param x: int
    :param y: int
    :param n: int
    :param m: int
    :return: list
    """
    bin_string = ''.join(['1' for _ in range(x + y + n + m)])
    int_ = int(bin_string, 2)
    res = bin(int(int_, 10))
    return res


def range_transfer(x, max_border, min_border=0):
    """
    边界转换：将x属于[-1,1]转换为 max, min border的树枝
    :param x: ndarray(float)
    :param max_border: int
    :param min_border: int
    :return: ndarray(in 属于 [min, max]
    """
    max_border -= 1  # 0-max_border-1
    ret = (max_border * (np.exp(x + 1) - 1)) / (np.exp(2) - 1)
    ret = np.int(np.rint(ret))  # rint: 四舍五入
    return ret


def action2direction(direct, act, length):
    """
    将动作转换为偏角
    :param direct: numpy.int64
    :param act: numpy.int64
    :param length: int
    :return: direction: numpy.int64
    """
    mid_length = int(length / 2)
    if act <= mid_length:
        # 向下偏移
        direction = (direct + act) % 360
    else:
        # 向上偏移
        deflection = act - mid_length
        if direct > deflection:
            direction = direct - deflection
        else:
            direction = 360 + direct - deflection
    # print(type(direction))
    # assert(isinstance(direction, np.int64))
    return direction


def action2direction2(direct, act):
    """
    将动作转换为偏角
    :param direct: numpy.int64
    :param act: numpy.int64
    :param length: int
    :return: direction: numpy.int64
    """
    deflection_list = [0, -15, -30, -45, 15, 30, 45]
    deflection = deflection_list[act]
    direction = direct + deflection
    if direction >= 360:
        direction %= 360
    elif direction < 0:
        direction += 360
    # print(type(direction))
    # assert(isinstance(direction, np.int64))
    return direction


def norm(x):
    """
    归一化
    :param x: tensor
    :return: ret: tensor
    """
    x_max = th.max(x)
    x_min = th.min(x)
    ret = (x - x_min) / (x_max - x_min)
    return ret


def norm_np(x):
    """
    归一化
    :param x: ndarray
    :return: ret: ndarray
    """
    x_max = np.max(x)
    x_min = np.min(x)
    ret = (x - x_min) / (x_max - x_min)
    return ret


def std(x):
    """
    :param x: tensor
    :return: ret: tensor
    """
    mu = th.mean(x)
    std = th.std(x)
    ret = (x - mu) / std
    return ret


def reward_w(reward, file_name):
    """
    将得分写入文件
    :param reward: float
    :param file_name: string
    :return:
    """
    score_string = '%.2f,' % reward
    with open(file_name, 'a') as f:
        f.write(score_string)


def reward_r(file_name):
    """
    :param file_name: string
    :return: list(float)
    """
    with open(file_name, 'r') as f:
        reward_string = f.read()
    reward_list = [float(ele) for ele in reward_string.split(',')[:-1]]
    return reward_list


def loss_w(loss, file_name):
    """
    :param loss: float
    :param file_name: string
    :return:
    """
    loss_string = '%.2f,' % loss
    with open(file_name, 'a') as f:
        f.write(loss_string)


def loss_r(file_name):
    """
    :param file_name: string
    :return: list(float)
    """
    with open(file_name, 'r') as f:
        loss_string = f.read()
    loss_list = [float(ele) for ele in loss_string.split(',')[:-1]]
    return loss_list


def make_pic(lst, title_name):
    """
    :param lst: list(float)
    :param title_name: string
    :return:
    """
    x = np.arange(1, len(lst) + 1)
    print("length: {}".format(len(lst)))
    y = np.array(lst)
    plt.title(title_name)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    # 创建解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--pics', type=str, default='', help='input sth')
    args = parser.parse_args()

    # reward
    if args.pics == 'reward':
        lst = reward_r('train/{}/pics/reward.txt'.format(PICS_PATH))
        make_pic(lst, 'reward')
    # loss
    elif args.pics == 'loss':
        lst = loss_r('train/dqn/pics/loss.txt')
        make_pic(lst, 'loss')
    elif args.pics == 'a_loss':
        lst = loss_r('train/{}/pics/agent0_a_loss.txt'.format(PICS_PATH))
        make_pic(lst, 'a loss')
    elif args.pics == 'c_loss':
        lst = loss_r('train/{}/pics/agent0_c_loss.txt'.format(PICS_PATH))
        make_pic(lst, 'c loss')
    # other utils test
    else:
        # print(action2direction(180, 0, 21))
        print(action2direction2(335, 6))

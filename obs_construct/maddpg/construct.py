"""
添加一个观测目标距离r_visible_dis 的obs
"""
import numpy as np
import copy
import config
from interface import get_distance


class ObsConstruct:
    def __init__(self, size_x, size_y, detector_num, fighter_num):
        self.battlefield_size_x = size_x
        self.battlefield_size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num

    def obs_construct(self, obs_raw_dict):
        obs_dict = {}
        detector_obs_list = []
        fighter_obs_list = []
        detector_data_obs_list = obs_raw_dict['detector_obs_list']
        fighter_data_obs_list = obs_raw_dict['fighter_obs_list']
        joint_data_obs_dict = obs_raw_dict['joint_obs_dict']

        detector_course, fighter_course = self.__get_course(detector_data_obs_list, fighter_data_obs_list)
        detector_pos, fighter_pos = self.__get_pos(detector_data_obs_list, fighter_data_obs_list)
        fighter_l_missile, fighter_s_missile = self.__get_missile(fighter_data_obs_list)
        detector_alive_status, fighter_alive_status = self.__get_alive_status(detector_data_obs_list,
                                                                              fighter_data_obs_list)
        detector_r_visible_pos, fighter_r_visible_pos, detector_r_visible_dis, fighter_r_visible_dis = self.__get_r_visible(
            detector_data_obs_list, fighter_data_obs_list)
        detector_j_visible_fp, fighter_j_visible_fp = self.__get_j_visible_fp(detector_data_obs_list,
                                                                              fighter_data_obs_list)
        detector_j_visible_dir, fighter_j_visible_dir = self.__get_j_visible_dir(detector_data_obs_list,
                                                                                 fighter_data_obs_list)
        fighter_striking_list = self.__get_striking_list(fighter_data_obs_list)
        joint_striking_pos = self.__get_joint_striking_pos(fighter_data_obs_list, joint_data_obs_dict)

        # 预警机
        for x in range(self.detector_num):
            course = detector_course[x, :]
            pos = detector_pos[x, :]
            alive_status = detector_alive_status[x][0]
            r_visible_pos = detector_r_visible_pos[x, :]
            r_visible_dis = detector_r_visible_dis[x, :]
            j_visible_fp = detector_j_visible_fp[x, :]
            j_visible_dir = detector_j_visible_dir[x, :]

            detector_obs_list.append({'course': copy.deepcopy(course), 'pos': copy.deepcopy(pos),
                                      'alive_status': alive_status[x][0],
                                      'r_visible_pos': copy.deepcopy(r_visible_pos),
                                      'r_visible_dis': copy.deepcopy(r_visible_dis),
                                      'j_visible_fp': copy.deepcopy(j_visible_fp),
                                      'j_visible_dir': copy.deepcopy(j_visible_dir)})
        # 战机
        for x in range(self.fighter_num):
            course = fighter_course[x, :]
            pos = fighter_pos[x, :]
            l_missile = fighter_l_missile[x, :]
            s_missile = fighter_s_missile[x, :]
            alive_status = fighter_alive_status[x, :]
            r_visible_pos = fighter_r_visible_pos[x, :]
            r_visible_dis = fighter_r_visible_dis[x, :]
            j_visible_fp = fighter_j_visible_fp[x, :]
            j_visible_dir = fighter_j_visible_dir[x, :]
            striking_id = fighter_striking_list[x, :]
            g_striking_pos = joint_striking_pos[0, :]

            fighter_obs_list.append({'course': copy.deepcopy(course), 'pos': copy.deepcopy(pos),
                                     'l_missile': copy.deepcopy(l_missile), 's_missile': copy.deepcopy(s_missile),
                                     'alive': copy.deepcopy(alive_status),
                                     'r_visible_pos': copy.deepcopy(r_visible_pos),
                                     'r_visible_dis': copy.deepcopy(r_visible_dis),
                                     'j_visible_fp': copy.deepcopy(j_visible_fp),
                                     'j_visible_dir': copy.deepcopy(j_visible_dir),
                                     'striking_id': copy.deepcopy(striking_id),
                                     'g_striking_pos': copy.deepcopy(g_striking_pos)})
        obs_dict['detector'] = detector_obs_list
        obs_dict['fighter'] = fighter_obs_list
        return obs_dict

    def __get_course(self, detector_data_obs_list, fighter_data_obs_list):
        detector_course = np.full((self.detector_num, 1), -1, dtype=np.int32)
        fighter_course = np.full((self.fighter_num, 1), -1, dtype=np.int32)
        for x in range(self.detector_num):
            if detector_data_obs_list[x]['alive']:
                detector_course[x, 0] = detector_data_obs_list[x]['course']
        for x in range(self.fighter_num):
            if fighter_data_obs_list[x]['alive']:
                fighter_course[x, 0] = fighter_data_obs_list[x]['course']

        return detector_course, fighter_course

    def __get_pos(self, detector_data_obs_list, fighter_data_obs_list):
        detector_pos = np.full((self.detector_num, 2), -1, dtype=np.int32)
        fighter_pos = np.full((self.fighter_num, 2), -1, dtype=np.int32)
        for x in range(self.detector_num):
            if detector_data_obs_list[x]['alive']:
                detector_pos[x, 0] = detector_data_obs_list[x]['pos_x']
                detector_pos[x, 1] = detector_data_obs_list[x]['pos_y']
        for x in range(self.fighter_num):
            if fighter_data_obs_list[x]['alive']:
                fighter_pos[x, 0] = fighter_data_obs_list[x]['pos_x']
                fighter_pos[x, 1] = fighter_data_obs_list[x]['pos_y']

        return detector_pos, fighter_pos

    def __get_alive_status(self, detector_data_obs_list, fighter_data_obs_list):
        detector_alive_status = np.full((self.detector_num, 1), True)
        fighter_alive_status = np.full((self.fighter_num, 1), True)
        # 探测单元　存活　从０　开始计数
        for x in range(self.detector_num):
            if not detector_data_obs_list[x]['alive']:
                detector_alive_status[x][0] = False
        # 战斗单元　存活　从０＋探测单元总数　开始计数
        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                fighter_alive_status[x][0] = False

        return detector_alive_status, fighter_alive_status

    def __get_missile(self, fighter_data_obs_list):
        fighter_l_missile = np.full((self.fighter_num, 1), -1, dtype=np.int32)
        fighter_s_missile = np.full((self.fighter_num, 1), -1, dtype=np.int32)
        for x in range(self.fighter_num):
            if fighter_data_obs_list[x]['alive']:
                fighter_l_missile[x, 0] = fighter_data_obs_list[x]['l_missile_left']
                fighter_s_missile[x, 0] = fighter_data_obs_list[x]['s_missile_left']

        return fighter_l_missile, fighter_s_missile

    def __get_r_visible(self, detector_data_obs_list, fighter_data_obs_list):
        detector_r_visible_pos = np.full((self.detector_num, 10, 2), -1,
                                         dtype=np.int32)  # todo : fighter_num + detector_num(12)
        fighter_r_visible_pos = np.full((self.fighter_num, 10, 2), -1,
                                        dtype=np.int32)  # todo : fighter_num + detector_num(12)
        detector_r_visible_dis = np.full((self.detector_num, 10, 1), -1,
                                         dtype=np.int32)  # todo : fighter_num + detector_num(12)
        fighter_r_visible_dis = np.full((self.fighter_num, 10, 1), 5000,
                                        dtype=np.int32)  # todo : fighter_num + detector_num(12)
        detector_pos, fighter_pos = self.__get_pos(detector_data_obs_list, fighter_data_obs_list)

        for x in range(self.detector_num):
            if detector_data_obs_list[x]['alive']:
                for y in range(len(detector_data_obs_list[x]['r_visible_list'])):
                    r_id = detector_data_obs_list[x]['r_visible_list'][y]['id']
                    detector_r_visible_pos[x, r_id - 1, 0] = detector_data_obs_list[x]['r_visible_list'][y]['pos_x']
                    detector_r_visible_pos[x, r_id - 1, 1] = detector_data_obs_list[x]['r_visible_list'][y]['pos_y']
                    detector_r_visible_dis[x, r_id - 1, 0] = get_distance(detector_pos[x, 0], detector_pos[x, 1],
                                                                       detector_r_visible_pos[x, r_id - 1, 0],
                                                                       detector_r_visible_pos[x, r_id - 1, 1])

        for x in range(self.fighter_num):
            if fighter_data_obs_list[x]['alive']:
                for y in range(len(fighter_data_obs_list[x]['r_visible_list'])):
                    r_id = fighter_data_obs_list[x]['r_visible_list'][y]['id']
                    fighter_r_visible_pos[x, r_id - 1, 0] = fighter_data_obs_list[x]['r_visible_list'][y]['pos_x']
                    fighter_r_visible_pos[x, r_id - 1, 1] = fighter_data_obs_list[x]['r_visible_list'][y]['pos_y']
                    fighter_r_visible_dis[x, r_id - 1, 0] = get_distance(fighter_pos[x, 0], fighter_pos[x, 1],
                                                                      fighter_r_visible_pos[x, r_id - 1, 0],
                                                                      fighter_r_visible_pos[x, r_id - 1, 1])

        return detector_r_visible_pos, fighter_r_visible_pos, detector_r_visible_dis, fighter_r_visible_dis

    def __get_j_visible_fp(self, detector_data_obs_list, fighter_data_obs_list):
        detector_j_visible_fp = np.full((self.detector_num, 10, 1), -1,
                                        dtype=np.int32)  # todo : fighter_num + detector_num(12)
        fighter_j_visible_fp = np.full((self.fighter_num, 10, 1), -1,
                                       dtype=np.int32)  # todo : fighter_num + detector_num(12)
        for x in range(self.detector_num):
            if detector_data_obs_list[x]['alive']:
                for y in range(len(detector_data_obs_list[x]['j_recv_list'])):
                    j_id = detector_data_obs_list[x]['j_recv_list'][y]['id']
                    detector_j_visible_fp[x, j_id - 1, 0] = detector_data_obs_list[x]['j_recv_list'][y]['r_fp']

        for x in range(self.fighter_num):
            if fighter_data_obs_list[x]['alive']:
                for y in range(len(fighter_data_obs_list[x]['j_recv_list'])):
                    j_id = fighter_data_obs_list[x]['j_recv_list'][y]['id']
                    fighter_j_visible_fp[x, j_id - 1, 0] = fighter_data_obs_list[x]['j_recv_list'][y]['r_fp']

        return detector_j_visible_fp, fighter_j_visible_fp

    def __get_j_visible_dir(self, detector_data_obs_list, fighter_data_obs_list):
        detector_j_visible_dir = np.full((self.detector_num, 10, 1), -1,
                                         dtype=np.int32)  # todo : fighter_num + detector_num(12)
        fighter_j_visible_dir = np.full((self.fighter_num, 10, 1), -1,
                                        dtype=np.int32)  # todo : fighter_num + detector_num(12)
        for x in range(self.detector_num):
            if detector_data_obs_list[x]['alive']:
                for y in range(len(detector_data_obs_list[x]['j_recv_list'])):
                    j_id = detector_data_obs_list[x]['j_recv_list'][y]['id']
                    detector_j_visible_dir[x, j_id - 1, 0] = detector_data_obs_list[x]['j_recv_list'][y]['direction']

        for x in range(self.fighter_num):
            if fighter_data_obs_list[x]['alive']:
                for y in range(len(fighter_data_obs_list[x]['j_recv_list'])):
                    j_id = fighter_data_obs_list[x]['j_recv_list'][y]['id']
                    fighter_j_visible_dir[x, j_id - 1, 0] = fighter_data_obs_list[x]['j_recv_list'][y]['direction']

        return detector_j_visible_dir, fighter_j_visible_dir

    def __get_joint_visible_pos(self, joint_data_obs_dict):
        g_visible_pos = np.full((1, 10, 2), -1, dtype=np.int32)  # todo : fighter_num + detector_num(12)
        for y in range(len(joint_data_obs_dict['passive_detection_enemy_list'])):
            g_visible_pos[0, 0] = joint_data_obs_dict['passive_detection_enemy_list'][y]['pos_x']
            g_visible_pos[0, 1] = joint_data_obs_dict['passive_detection_enemy_list'][y]['pos_y']

        return g_visible_pos

    def __get_striking_list(self, fighter_data_obs_list):
        fighter_striking_list = np.full((self.fighter_num, 10, 1), 0,
                                        dtype=np.int32)  # todo : fighter_num + detector_num(12)
        for x in range(self.fighter_num):
            if fighter_data_obs_list[x]['alive']:
                for y in range(len(fighter_data_obs_list[x]['striking_list'])):
                    if fighter_data_obs_list[x]['striking_list']:
                        j_id = fighter_data_obs_list[x]['striking_list'][y]
                        fighter_striking_list[x, j_id - 1, 0] = 1

        return fighter_striking_list

    def __get_joint_striking_pos(self, fighter_data_obs_list, joint_data_obs_dict):
        # detector_pos, fighter_pos = self.__get_pos(self, detector_data_obs_list, fighter_data_obs_list)
        target_fighter_pos = np.full((1, 10, 2), -1, dtype=np.int32)
        for y in range(len(joint_data_obs_dict['strike_list'])):
            attacker_id = joint_data_obs_dict['strike_list'][y]['attacker_id']
            target_id = joint_data_obs_dict['strike_list'][y]['target_id']
            for x in range(len(fighter_data_obs_list[attacker_id - 1]['striking_dict_list'])):
                if fighter_data_obs_list[attacker_id - 1]['striking_dict_list'][x]['target_id'] == target_id:
                    target_fighter_pos[0, target_id - 1, 0] = \
                        fighter_data_obs_list[attacker_id - 1]['striking_dict_list'][x]['pos_x']
                    target_fighter_pos[0, target_id - 1, 1] = \
                        fighter_data_obs_list[attacker_id - 1]['striking_dict_list'][x]['pos_y']

        return target_fighter_pos

#
# coding=utf-8


import os
import torch as th


class Logger(object):
    def __init__(self, level):
        if level == 'info':
            self.level = 1
        elif level == 'debug':
            self.level = 0
        elif level == 'danger':
            self.level = 2
        else:
            raise (Exception('level params error!'))

    def info(self, msg):
        if self.level < 2:
            print(msg)

    def debug(self, msg=None):
        if self.level < 1:
            msg = msg if msg else 'wait!'
            print(msg)
            # input('pause: input sth and enter: ')

    def warn(self, msg):
        if self.level < 3:
            print(msg)

    def wait(self):
        input('wait! pause: input sth and enter: ')


# 日志单例
logger = Logger(level="info")


class GPUConfig(object):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # todo 一定要放在th.cuda前面
        self.use_cuda = False  # th.cuda.is_available()
        self.use_parallel = False
        self.device_ids = [0, 1]
        if self.use_cuda:
            pass
            # 分配gpu
            # th.cuda.set_device(1)
            # self.device = th.device("cuda:0")  # 指定模型训练所在 GPU
            if self.use_parallel:
                th.distributed.init_process_group(backend='nccl')
                # python -m torch.distributed.launch main.py


GPU_CONFIG = GPUConfig()

# model agent name
MODEL_NAME = 'model_000035050_agent'
# MODEL_NAME_AGENT = 'model_000029900_agent'  # maddpg

# model store path
MODEL_PATH = 'maddpg'

# model agent path
# MODEL_PATH_AGENT = 'dqn_sjy'

# pics path or ''
PICS_PATH = 'maddpg'

# todo is test
IS_TEST = True

# 离散动作
IS_DISPERSED = True

if __name__ == '__main__':
    # logger.debug("debug")
    pass

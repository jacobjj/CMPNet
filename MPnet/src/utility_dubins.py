import torch
from torch.autograd import Variable
import copy
import numpy as np
import time


def normalize(x, bound, time_flag=False):
    """
    Normalize the environment coordinates between -1 and 1. Since the environment is symmetric we divide by range/2
    """
    time_0 = time.time()
    bound = torch.tensor(bound)
    if x.shape[1] == len(bound):
        x = x / bound
    else:
        x[:, :3] = normalize(x[:, :3], bound)
        x[:, 3:] = normalize(x[:, :3], bound)
    if time_flag:
        return x, time.time() - time_0
    return x


def denormalize(x, bound, time_flag=False):
    """
    Denoralize the environment co-ordinates between -bound and bound. Since the environment is symmetric we multiply by range/2.
    """
    time_0 = time.time()
    bound = torch.tensor(bound)
    if x.shape[1] == len(bound):
        x = x * bound
    else:
        x[:, :3] = denormalize(x[:, :3], bound)
        x[:, :3] = denormalize(x[:, :3], bound)
    if time_flag:
        return x, time.time() - time_0
    return x

"""List of common functions used for this library"""
import torch
import numpy as np


def normalize_angle(z):
    """
    A function to wrap around -1 and 1
    """
    return (z + np.pi) % (2 * np.pi) - np.pi


def LeftTurn(x, beta):
    return torch.Tensor([
        x[0] + torch.sin(x[2] + beta) - torch.sin(x[2]),
        x[1] - torch.cos(x[2] + beta) + torch.cos(x[2]),
        normalize_angle(beta + x[2]),
    ])


def RightTurn(x, alpha):
    return torch.Tensor([
        x[0] - torch.sin(x[2] - alpha) + torch.sin(x[2]),
        x[1] + torch.cos(x[2] - alpha) - torch.cos(x[2]),
        normalize_angle(alpha - x[2]),
    ])


def StraightTrun(x, gamma):
    return torch.Tensor([
        x[0] + gamma * torch.cos(x[2]),
        x[1] + gamma * torch.sin(x[2]),
        x[2],
    ])


def word2primitive(word, p, start, d):
    """
    Converts the word back to the primitive path
    Define the word as follows:
    index 0 - RightTurn
    index 1 - LeftTurn
    index 2 - Straight
    Each row has the following representation
    LSL = 0
    LSR = 1
    RSL = 2
    RSR = 3
    RLR = 4
    LRL = 5
    """
    F = [RightTurn, LeftTurn, StraightTrun]
    for i in [0,1,3,4,6,7]:
        word[i] = word[i] * np.pi
    word = word.reshape((3, 3))
    p = p.reshape((3, 3))
    for i in range(3):
        _, index = torch.max(p[i, :], dim=0)
        v = word[i, index]
        if index == 0 or index == 1:
            start[:2] = start[:2] / d
        start = F[index](start, v)
        if index == 0 or index == 1:
            start[:2] = start[:2] * d
    return start

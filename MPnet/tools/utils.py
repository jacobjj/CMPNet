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
        normalize_angle(-alpha + x[2]),
    ])


def StraightTurn(x, gamma):
    return torch.Tensor([
        x[0] + gamma * torch.cos(x[2]),
        x[1] + gamma * torch.sin(x[2]),
        x[2],
    ])


def word2primitive(word, start, d):
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
    F = [RightTurn, LeftTurn, StraightTurn]
    temp_word = torch.abs(word.clone())
    for i in [0, 1, 3, 4, 6, 7]:
        temp_word[i] = temp_word[i] * np.pi * d
        word[i] = word[i] * np.pi
    word = word.reshape((3, 3))
    temp_word = temp_word.reshape((3, 3))
    x_i = torch.Tensor([0, 0, start[2]])
    for i in range(3):
        _, index = torch.max(temp_word[i, :], dim=0)
        v = word[i, index]
        if index == 2:
            v = v / d
        x_i = F[index](x_i, v)
    start[0] += x_i[0] * d
    start[1] += x_i[1] * d
    start[2] = normalize_angle(start[2] + x_i[2])
    return start

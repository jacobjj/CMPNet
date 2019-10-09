""" A nn parameterized by dubins curve"""
import torch.nn as nn
import torch
import numpy as np


def normalize_angle(z):
    """
    A function to wrap around -1 and 1
    """
    return (z + 1) % (2) - 1


class DubinsPathGenerator(nn.Module):
    def __init__(self, input_size):
        super(DubinsPathGenerator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size + 3, 64),
            nn.PReLU(),
            nn.Linear(64, 3),
        )

    def LeftTurn(self, x, beta):
        return torch.Tensor([
            x[:, 0] + torch.sin(x[:, 2] + beta) - torch.sin(x[:, 2]),
            x[:, 1] - torch.cos(x[:, 2] + beta) + torch.cos(x[:, 2]),
            normalize_angle(beta + x[:, 2]),
        ])

    def RightTurn(self, x, alpha):
        return torch.Tensor([
            x[:, 0] - torch.sin(x[:, 2] - alpha) + torch.sin(x[:, 2]),
            x[:, 1] + torch.cos(x[:, 2] - alpha) - torch.cos(x[:, 2]),
            normalize_angle(alpha - x[:, 2]),
        ])

    def StraightTrun(self, x, gamma):
        return torch.Tensor([
            x[:, 0] + gamma * torch.cos(x[:, 2]),
            x[:, 1] + gamma * torch.sin(x[:, 2]),
            x[:, 2],
        ])

    def forward(self, c):
        hidden = torch.zeros(c.shape[0], 3).cuda()
        s = []
        for i in range(3):
            concat = torch.cat((c, hidden), 1)
            hidden = self.fc(concat)
            s.append(hidden)
        return torch.cat(s,dim=1)

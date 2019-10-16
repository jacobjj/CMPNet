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
            nn.Linear(input_size + 3*2, 64),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(64, 64),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Dropout(),
        )
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.final1 = nn.Linear(32, 3)
        self.final2 = nn.Linear(32, 3)

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
        hidden = torch.cat((torch.zeros(
            c.shape[0], 3).cuda(), torch.ones(c.shape[0], 3).cuda() / 3),
                           dim=1)

        s = []
        p_s = []
        for i in range(3):
            concat = torch.cat((c, hidden), 1)
            hidden = self.fc(concat)
            hidden1 = self.final1(hidden)
            hidden1 = torch.cat(
                (self.tanh(hidden1[:, :2]), self.relu(hidden1[:, 2]).reshape(
                    (-1, 1))),
                dim=1)
            hidden_p = self.final2(hidden)
            hidden_p = self.softmax(hidden_p)
            s.append(hidden1)
            p_s.append(hidden_p)
            hidden = torch.cat((hidden1, hidden_p),dim=1)
        return [torch.cat(s, dim=1), torch.cat(p_s, dim=1)]

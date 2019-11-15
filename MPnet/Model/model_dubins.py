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
            # nn.Linear(input_size , 64),
            # nn.PReLU(),
            # nn.Dropout(),
            # nn.Linear(64, 64),
            # nn.PReLU(),
            # nn.Dropout(),
            # nn.Linear(64, 32),
            # nn.PReLU(),
            # nn.Dropout(),
            nn.Linear(input_size, 128),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Dropout(),
        )
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.final = nn.Linear(32, 7)

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
        x = self.fc(c)
        x = self.final(x)
        filler = torch.zeros((c.shape[0], 1), dtype=c.dtype, device=c.device)
        x = torch.cat(
            (
                self.tanh(x[:, :2]),
                # self.relu(x[:, 2]).reshape((-1, 1)),
                filler,
                self.tanh(x[:, 2:4]),
                self.relu(x[:, 4]).reshape((-1, 1)),
                self.tanh(x[:, 5:7]),
                # self.relu(x[:, 8]).reshape((-1, 1))
                filler,
            ),
            dim=1,
        )
        return x

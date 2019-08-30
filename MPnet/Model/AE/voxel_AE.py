import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """
    A 2D VoxNet for encoding obstacle space
    """

    def __init__(self, output_size, input_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=8,
                      kernel_size=[3, 3],
                      stride=[1, 1]),
            nn.PReLU(),
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=[5, 5],
                      stride=[1, 1]),
            nn.PReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=[5, 5],
                      stride=[1, 1]),
            nn.PReLU(),
        )

        # For accepting different input shapes
        x = self.encoder(torch.autograd.Variable(torch.rand([1] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features, 128),
            nn.PReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

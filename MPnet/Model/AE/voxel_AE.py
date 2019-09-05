import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # nn.init.xavier_normal_(m.weight.data)
        m.weight.data.normal_(0.0, 1e-3)
    elif classname.find('Linear') != -1:
        # nn.init.xavier_normal_(m.weight.data)
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0)


class Encoder(nn.Module):
    """
    A 2D VoxNet for encoding obstacle space
    """

    def __init__(self, output_size, input_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=8,
                      kernel_size=[5, 5],
                      stride=[1, 1]),
            # nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=3),
            nn.PReLU(),
            nn.Conv2d(in_channels=8,
                      out_channels=4,
                      kernel_size=[3, 3],
                      stride=[1, 1]),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3),
            nn.PReLU(),
            # nn.Conv2d(in_channels=16,
            #           out_channels=32,
            #           kernel_size=[3, 3],
            #           stride=[1, 1]),
            # # nn.BatchNorm2d(32),
            # nn.PReLU(),
        )
        self.encoder.apply(weights_init)
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
        self.head.apply(weights_init)
        self.set_clip_grad = True

    def get_contractive_loss(self):
        """
        Return contractive loss for the outer layer
        """
        keys = list(self.head.state_dict().keys())
        W = Variable(self.head.state_dict()[keys[-2]])
        if torch.cuda.is_available():
            W = W.cuda()
        contractive_loss = torch.sum(W**2, dim=1).sum()
        return contractive_loss

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


def clip_grad(v, min, max):
    """
    A way to clip the gradients
    From - https://github.com/DingKe/pytorch_workplace/blob/master/rnn/modules.py
    """
    v.register_hook(lambda g: g.clamp(min, max))
    return v

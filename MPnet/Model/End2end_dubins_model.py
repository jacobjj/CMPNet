import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal
from Model.gem_utility import *
import numpy as np
import copy
import dubins
from torch.nn.utils import clip_grad_norm_

get_numpy = lambda x: x.data.cpu().numpy()


def normalize_cost(z):
    """
    A function to wrap around -1 and 1
    """
    return (z + 1) % 2 - 1


# Auxiliary functions useful for GEM's inner optimization.
class End2EndMPNet(nn.Module):
    """ A python class defining the components of MPnet"""

    def __init__(
            self,
            AE_input_size,
            AE_output_size,
            state_size,
            mlp_input_size,
            n_tasks,
            grad_step,
            CAE,
            MLP,
    ):
        """
        TODO : update older generation code to be compatible with this class
        : param total_input_size :
        : param AE_input_size :
        : param mlp_input_size :
        : param output_size :
        : param AEtype :
        : param n_tasks :
        : param n_memories :
        : param memory_strength :
        : param grad_step :
        : param CAE :
        : param MLP :
        """
        super(End2EndMPNet, self).__init__()
        self.encoder = CAE.Encoder(AE_output_size, state_size, AE_input_size)
        self.mlp = MLP(mlp_input_size)
        self.mse = nn.MSELoss()
        self.set_opt(torch.optim.Adagrad, lr=1e-4)

        self.num_seen = np.zeros(n_tasks).astype(int)
        self.grad_step = grad_step
        self.AE_input_size = AE_input_size
        # Remove any gradient set during initialization
        self.zero_grad()

    def set_opt(self, opt, lr=1e-2, momentum=None):
        # edit: can change optimizer type when setting
        if momentum is None:
            self.opt = opt(list(self.encoder.parameters()) +
                           list(self.mlp.parameters()),
                           lr=lr)
        else:
            self.opt = opt(list(self.encoder.parameters()) +
                           list(self.mlp.parameters()),
                           lr=lr,
                           momentum=momentum)

    def forward(self, x, obs):
        """
        Forward step of MPnet
        : param obs : input to encoder
        : param x : input to mlp
        """
        c = self.encoder(obs, x)
        return self.mlp(c)

    def get_path_length(self, startNode, endNode):
        """
        A function to generate dubins path object
        :param startNode : A tuple indicating the start position
        :param endNode : A tuple indicating goal position
        : returns (scalar) : path length of the shortest dubins curve.
        """
        d = 0.6
        try:
            path = dubins.shortest_path(startNode, endNode, d)
            return path.path_length()
        except RuntimeError:
            import pdb
            pdb.set_trace()

    def dubins_path_loss(self, x, y):
        """
        A function that estimates the dubins curve distance from x to y.
        """
        dubins_path_distance = []
        for x_i, y_i in zip(x, y):
            d = self.get_path_length(tuple(get_numpy(x_i)),
                                     tuple(get_numpy(y_i)))
            dubins_path_distance.append(d)
        return torch.tensor(dubins_path_distance)

    def loss(self, pred, truth):
        # try:
        #     contractive_loss = self.encoder.get_contractive_loss()
        # except AttributeError:
        #     return self.mse(pred, truth)
        # NOTE: This cost function is designed for r2d cars and need to change to
        # be compatible with other methods
        try:
            loss_cord = self.mse(pred[:, 2], truth[:, 2])
            loss_angles = torch.mean(normalize_cost(pred[:, 0] - truth[:, 0])**2+normalize_cost(pred[:, 1] - truth[:, 1])**2)
        except:
            import pdb; pdb.set_trace()
            
        return loss_angles + loss_cord

    def loss_with_regularize(self, pred, truth):
        """
        Loss with regularization included.
        """
        loss = 0
        for i in range(3):
            loss += self.loss(pred[:, i * 3:(i + 1) * 3],
                              truth[:, i * 3:(i + 1) * 3])
        return loss

    def fit(self, obs, x, y):
        """
        Updates the network weights to best fit the given data.
        :param obs: the voxel representation of the obstacle space
        :param x: the state of the robot
        :param y: the next state of the robot
        NOTE: It is safer to call nn.Module.zero_grad() rather than optim.zero_grad(). If the encoder and decoder network has different optim functions, then this takes care for setting gradients of both model to zero.
        """
        loss = self.loss_with_regularize(self.__call__(x, obs), y)
        loss.backward()
        self.opt.step()
        self.zero_grad()

    def fit_distribution(self, obs, x, y):
        """
        Updates the network weights to generate the best distribution, that maximizes the dubins distance from the sampled point to the dubins curve.
        :param obs: the voxel representation of the obstacle space
        :param x: the state of the robot
        :param y: the next state of the robot
        """
        y_hat, log_prob_y_hat = self.sample(obs, x)
        distance = self.dubins_path_loss(y_hat, y)
        loss = torch.mean(log_prob_y_hat * distance)
        loss.backward()
        # grad_norm = clip_grad_norm_(self.encoder.parameters(),1.5)
        grad_norm = clip_grad_norm_(self.mlp.parameters(), 1.5)
        self.opt.step()
        self.zero_grad()
        return grad_norm

    def sample(self, obs, x):
        """
        A function that returns the sampled point along with its log probability
        """
        mean = self.forward(x, obs).cpu()
        m = MultivariateNormal(mean, self.covar_matrix)
        next_state = m.sample()
        log_prob = m.log_prob(next_state)
        return next_state, log_prob

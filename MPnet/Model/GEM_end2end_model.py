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
            mlp_input_size,
            mlp_output_size,
            AEtype,
            n_tasks,
            n_memories,
            memory_strength,
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
        self.encoder = CAE.Encoder(AE_output_size, AE_input_size)
        self.mlp = MLP(mlp_input_size, mlp_output_size)
        self.mse = nn.MSELoss()
        self.covar_matrix = torch.eye(mlp_output_size) * 0.1
        self.set_opt(torch.optim.Adagrad, lr=1e-4)
        '''
        Below is the attributes defined in GEM implementation
        reference: https://arxiv.org/abs/1706.08840
        code from: https://github.com/facebookresearch/GradientEpisodicMemory
        '''
        self.margin = memory_strength
        self.n_memories = n_memories
        # allocate episodic memory

        self.memory_data = torch.FloatTensor(n_tasks, self.n_memories,
                                             mlp_input_size)
        self.memory_obs = torch.FloatTensor(n_tasks, self.n_memories,
                                            np.prod(AE_input_size))
        self.memory_labs = torch.FloatTensor(n_tasks, self.n_memories,
                                             mlp_output_size)
        if torch.cuda.is_available():
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        # edit: need one more dimension for newly observed data
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks + 1)
        if torch.cuda.is_available():
            self.grads = self.grads.cuda()
        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = np.zeros(n_tasks).astype(int)
        self.num_seen = np.zeros(n_tasks).astype(int)
        self.grad_step = grad_step
        self.AE_input_size = AE_input_size
        # Remove any gradient set during initialization
        self.zero_grad()

    def clear_memory(self):
        # set the counter to 0
        self.mem_cnt[:] = 0
        # set observed task to empty
        self.observed_tasks = []
        self.old_task = -1

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
        z = self.encoder(obs)
        mlp_in = torch.cat((z, x), 1)
        return self.mlp(mlp_in)

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
            import pdb; pdb.set_trace()


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

        loss_cord = self.mse(pred[:, :2], truth[:, :2])
        loss_angles = torch.mean(normalize_cost(pred[:, 2] - truth[:, 2])**2)
        return loss_angles + loss_cord

    def loss_with_regularize(self, pred, truth):
        """
        Loss with regularization included.
        """
        loss = self.loss(pred, truth)
        contractive_loss = self.encoder.get_contractive_loss()
        return loss + contractive_loss * 0.01

    def load_memory(self, data):
        # data: (tasks, xs, ys)
        # continuously load memory based on previous memory loading
        tasks, xs, ys = data
        for i in range(len(tasks)):
            if tasks[i] != self.old_task:
                # new task, clear mem_cnt
                self.observed_tasks.append(tasks[i])
                self.old_task = tasks[i]
                self.mem_cnt[tasks[i]] = 0
            x = torch.tensor(xs[i])
            y = torch.tensor(ys[i])
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            self.remember(x, tasks[i], y)

    def remember(self, x, t, y):
        # follow reservoir sampling
        # i-th item is remembered with probability min(B/i, 1)
        for i in range(len(x)):
            self.num_seen[t] += 1
            prob_thre = min(self.n_memories, self.num_seen[t])
            rand_num = np.random.choice(self.num_seen[t],
                                        1)  # 0---self.num_seen[t]-1
            if rand_num < prob_thre:
                # keep the new item
                if self.mem_cnt[t] < self.n_memories:
                    self.memory_data[t, self.mem_cnt[t]].copy_(x.data[i])
                    self.memory_labs[t, self.mem_cnt[t]].copy_(y.data[i])
                    self.mem_cnt[t] += 1
                else:
                    # randomly choose one to rewrite
                    idx = np.random.choice(self.n_memories, size=1)
                    idx = idx[0]
                    self.memory_data[t, idx].copy_(x.data[i])
                    self.memory_labs[t, idx].copy_(y.data[i])

    def fit(self, obs, x, y):
        """
        Updates the network weights to best fit the given data.
        :param obs: the voxel representation of the obstacle space
        :param x: the state of the robot
        :param y: the next state of the robot
        NOTE: It is safer to call nn.Module.zero_grad() rather than optim.zero_grad(). If the encoder and decoder network has different optim functions, then this takes care for setting gradients of both model to zero.
        """
        loss = self.loss_with_regularize(self.forward(x, obs), y)
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
        grad_norm = clip_grad_norm_(self.mlp.parameters(),1.5)
        self.opt.step()
        self.zero_grad()
        return grad_norm

    '''
    Below is the added GEM feature
    reference: https://arxiv.org/abs/1706.08840
    code from: https://github.com/facebookresearch/GradientEpisodicMemory
    '''

    def observe(self, obs, x, t, y, remember=True):
        # remember: remember this data or not
        # update memory
        # everytime we treat the new data as a new task
        # compute gradient on all tasks
        # (prevent forgetting previous experience of same task, too)
        for _ in range(self.grad_step):
            if len(self.observed_tasks) >= 1:
                for tt in range(len(self.observed_tasks)):
                    if self.mem_cnt[tt] == 0 and tt == len(
                            self.observed_tasks) - 1:
                        # nothing to train on
                        continue
                    self.zero_grad()
                    # fwd/bwd on the examples in the memory
                    past_task = self.observed_tasks[tt]
                    if tt == len(self.observed_tasks) - 1:
                        ptloss = self.loss(
                            self.forward(
                                self.memory_obs[past_task]
                                [:self.mem_cnt[past_task]],
                                self.memory_data[past_task]
                                [:self.mem_cnt[past_task]]),  # TODO
                            self.memory_labs[past_task]
                            [:self.mem_cnt[past_task]])  # up to current
                    else:
                        ptloss = self.loss(
                            self.forward(self.memory_obs[past_task],
                                         self.memory_data[past_task]),  # TODO
                            self.memory_labs[past_task])
                    ptloss.backward()
                    store_grad(self.parameters, self.grads, self.grad_dims,
                               past_task)

            # now compute the grad on the current minibatch
            loss = self.loss(self.forward(x, obs), y)
            loss.backward()

            # check if gradient violates constraints
            # treat gradient of current data as a new task (max of observed task + 1)
            # just to give it a new task label
            if len(self.observed_tasks) >= 1:
                # copy gradient
                new_t = max(self.observed_tasks) + 1  # a new dimension
                store_grad(self.parameters, self.grads, self.grad_dims, new_t)
                indx = torch.cuda.LongTensor(self.observed_tasks) if torch.cuda.is_available() \
                    else torch.LongTensor(self.observed_tasks)   # here we need all observed tasks
                # here is different, we are using new_t instead of t to ditinguish between
                # newly observed data and previous data
                dotp = torch.mm(self.grads[:, new_t].unsqueeze(0),
                                self.grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    # remember norm
                    norm = torch.norm(self.grads[:, new_t], 2)
                    project2cone2(self.grads[:, new_t].unsqueeze(1),
                                  self.grads.index_select(1, indx),
                                  self.margin)
                    new_norm = torch.norm(self.grads[:, new_t], 2)
                    self.grads[:, new_t].copy_(self.grads[:, new_t] /
                                               new_norm * norm)
                    # before overwrite, to avoid gradient explosion, renormalize the gradient
                    # so that new gradient has the same l2 norm as previous
                    # it can be proved theoretically that this does not violate the non-negativity
                    # of inner product between memory and gradient
                    # copy gradients back
                    overwrite_grad(self.parameters, self.grads[:, new_t],
                                   self.grad_dims)
            self.opt.step()
            # It is safer to call nn.Module.zero_grad() rather than optim.zero_grad() since this is the safer option. If the encoder and decoder network has different optim functions, then this takes care for setting gradients of both model to zero.
            self.zero_grad()
        # after training, store into memory

        # when storing into memory, we use the correct task label
        # Update ring buffer storing examples from current task
        if remember:
            # only remember when the flag is TRUE
            if t != self.old_task:
                # new task, clear mem_cnt
                self.observed_tasks.append(t)
                self.old_task = t
                self.mem_cnt[t] = 0
            self.remember(x, t, y)

    def sample(self, obs, x):
        """
        A function that returns the sampled point along with its log probability
        """
        mean = self.forward(x, obs).cpu()
        m = MultivariateNormal(mean, self.covar_matrix)
        next_state = m.sample()
        log_prob = m.log_prob(next_state)
        return next_state, log_prob

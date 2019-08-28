from Model.GEM_end2end_model import End2EndMPNet
from utility import load_net_state, load_opt_state, save_state, to_var

import data_loader_dubins
import numpy as np
import os
import os.path as osp
import random
import re
import torch
import csv

get_numpy = lambda x: x.data.cpu().numpy()


class MPnetBase():
    """
    MPnet implementation for path planning
    """

    def __init__(self,
                 normalize,
                 denormalize,
                 encoderInputDim,
                 encoderOutputDim,
                 worldSize,
                 n_tasks=1,
                 n_memories=256,
                 memory_strength=0.5,
                 grad_steps=1,
                 learning_rate=0.1,
                 AE=None,
                 MLP=None,
                 load_dataset=None,
                 modelPath=None):
        """
        Initialize the mpnet planner
        : param IsInCollsion
        : param normalize
        : param denormalize
        : param CAE
        : param MLP
        : param worldsize
        : param n_tasks,
        : param n_memories
        : param memory_strength
        : param grad_steps : number of gradient descent steps taken for optimizing MPNet in each epoch
        : param learning_rate
        : param n_epochs
        """
        self.torch_seed = np.random.randint(low=0, high=1000)
        self.np_seed = np.random.randint(low=0, high=1000)
        self.py_seed = np.random.randint(low=0, high=1000)
        torch.manual_seed(self.torch_seed)
        np.random.seed(self.np_seed)
        random.seed(self.py_seed)
        if not (AE or MLP):
            raise NotImplementedError("Add autoencoder and MLP network")
        self.normalize = normalize
        self.denormalize = denormalize
        self.worldSize = worldSize
        self.worldInputDim = len(worldSize)
        self.AE = AE
        self.MLP = MLP
        if not load_dataset:
            self.load_dataset = data_loader_dubins.load_dataset
        else:
            self.load_dataset = load_dataset
        self.mpNet = End2EndMPNet(
            total_input_size=self.worldInputDim * 2 + encoderInputDim,
            AE_input_size=encoderInputDim,
            mlp_input_size=self.worldInputDim * 2 + encoderOutputDim,
            output_size=self.worldInputDim,
            AEtype='deep',
            n_tasks=n_tasks,
            n_memories=n_memories,
            memory_strength=memory_strength,
            grad_step=grad_steps,
            CAE=self.AE,
            MLP=self.AE,
        )
        if torch.cuda.is_available():
            self.mpNet.cuda()
            self.mpNet.mlp.cuda()
            self.mpNet.encoder.cuda()

        self.mpNet.set_opt(torch.optim.Adagrad, lr=learning_rate)
        self.modelPath = modelPath

    def load_network_parameters(self, modelFile):
        """
        A method to load previously trained network parameters
        : param modelPath : location of the model parameters of the model
        """
        load_opt_state(self.mpNet, modelFile)

    def save_network_state(self, fname):
        """
        A method to save states of the network
        """
        save_state(self.mpNet, self.torch_seed, self.np_seed, self.py_seed,
                   fname)

    def format_data(self, obs, inputs, targets):
        """
        Formats the data to be fed into the neural network
        """
        bi = np.concatenate((obs, inputs), axis=1).astype(np.float32)
        bi = torch.FloatTensor(bi)
        bt = torch.FloatTensor(targets)
        bi, bt = self.normalize(bi, self.worldSize), self.normalize(
            bt, self.worldSize)
        bi = to_var(bi)
        bt = to_var(bt)
        return bi, bt

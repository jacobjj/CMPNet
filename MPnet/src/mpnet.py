from Model.GEM_end2end_model import End2EndMPNet
from src.utility import load_net_state, load_opt_state, save_state, to_var

import src.data_loader_dubins
import numpy as np
import os
import os.path as osp
import random
import re
import torch
import csv
import datetime
import torchvision

get_numpy = lambda x: x.data.cpu().numpy()


def generateModelPath():
    newDir = osp.join(osp.getcwd(),
                      datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    return newDir


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

        self.mpNet = End2EndMPNet(
            AE_input_size=encoderInputDim,
            AE_output_size=encoderOutputDim,
            mlp_input_size=self.worldInputDim * 2 + encoderOutputDim,
            mlp_output_size=self.worldInputDim,
            AEtype='deep',
            n_tasks=n_tasks,
            n_memories=n_memories,
            memory_strength=memory_strength,
            grad_step=grad_steps,
            CAE=AE,
            MLP=MLP,
        )
        if torch.cuda.is_available():
            self.mpNet.cuda()
            self.mpNet.mlp.cuda()
            self.mpNet.encoder.cuda()

        self.mpNet.set_opt(torch.optim.Adagrad, lr=learning_rate)
        # TODO: check if model path exists, otherwise create a new folder
        if modelPath == None:
            modelPath = generateModelPath()

        if osp.exists(modelPath):
            self.modelPath = modelPath
        else:
            raise ValueError("Not a valid directory")

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

    def format_input(self, obs, inputs):
        """
        Formats the input data that needed to be fed into the network
        """
        normObsVoxel = torchvision.transforms.Normalize([0.5], [1])
        bi = torch.FloatTensor(inputs)
        bobs = torch.FloatTensor(obs)
        # Normalize observations
        for i in range(bobs.shape[0]):
            bobs[i, ...] = normObsVoxel(bobs[i, ...])
        bi = self.normalize(bi, self.worldSize)
        return to_var(bobs), to_var(bi)

    def format_data(self, obs, inputs, targets):
        """
        Formats the data to be fed into the neural network
        """
        bobs, bi = self.format_input(obs, inputs)
        bt = torch.FloatTensor(targets)
        bt = self.normalize(bt, self.worldSize)
        bt = to_var(bt)
        return bobs, bi, bt

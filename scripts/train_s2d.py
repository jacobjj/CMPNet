import numpy as np
from src.data_loader_2d import load_dataset
import utility_s2d

import src.Model.AE.CAE as CAE_2d
import src.Model.model as model

from src.mpnet_trainv2 import MPnetTrain

from src.Model.model import model
import argparse


def train(args):
    normalize = utility_s2d.normalize
    denormalize = utility_s2d.unnormalize
    CAE = CAE_2d
    MLP = model.MLP

    network_parameters = {
        'normalize': normalize,
        'denormalize': denormalize,
        'encoderInputDim': 2800,
        'encoderOutputDim': 4,
        'worldSize': [20, 20],
        'AE': CAE_2d,
        'MLP': MLP,
        'modelPath': args.file,
        'load_dataset': load_dataset
    }

    trainNetwork = MPnetTrain(
        load_dataset,
        n_epochs=1000,
        batchSize=256,
        **network_parameters,
    )

    trainNetwork.train(numEnvs=100,
                       numPaths=1000,
                       trainDataPath='',
                       testDataPath='')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', tupe=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)

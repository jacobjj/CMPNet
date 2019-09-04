import numpy as np
from src.data_loader_2d import load_dataset
from src.data_loader_dubins import load_dataset_voxel

# import src.Model.AE.CAE as CAE_2d
import Model.AE.voxel_AE as voxelNet
import Model.model as model

from src.utility_r2d import normalize, unnormalize
from src.mpnet_trainv2 import MPnetTrain

import argparse


def train(args):
    denormalize = unnormalize
    MLP = model.MLP

    network_parameters = {
        'normalize': normalize,
        'denormalize': denormalize,
        'encoderInputDim': [1, 61, 61],
        'encoderOutputDim': 64,
        'worldSize': [2.75, 2.75, np.pi],
        'AE': voxelNet,
        'MLP': MLP,
        'modelPath': args.file,
    }

    trainNetwork = MPnetTrain(
        load_dataset=load_dataset_voxel,
        n_epochs=500,
        batchSize=256,
        **network_parameters,
    )

    trainNetwork.train(numEnvs=20000,
                       numPaths=1,
                       trainDataPath='data/dubinsCar',
                       testDataPath='data/dubinsCar_test')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)

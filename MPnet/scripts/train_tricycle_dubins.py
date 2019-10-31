import numpy as np
from src.data_loader_2d import load_dataset
from src.data_loader_dubins import load_dataset_voxel, load_dataset_torch

# import src.Model.AE.CAE as CAE_2d
import Model.AE.voxel_AE as voxelNet
import Model.model_dubins as model


from src.utility_r2d import normalize, unnormalize
from src.mpnet_trainv2 import MPnetTrain

import argparse
import torch


def train(args):
    denormalize = unnormalize
    MLP = model.DubinsPathGenerator

    network_parameters = {
        'normalize': normalize,
        'denormalize': denormalize,
        'encoderInputDim': [1, 122, 122],
        'encoderOutputDim': 128,
        'worldSize': [2.75, 2.75, np.pi],
        'AE': voxelNet,
        'MLP': MLP,
        'modelPath': args.file,
    }
    
    trainNetwork = MPnetTrain(
        load_dataset=load_dataset_torch,
        n_epochs=1000,
        batchSize=128,
        opt=torch.optim.Adagrad,
        # learning_rate=1e-5,
        **network_parameters,
    )
    # trainNetwork.set_model_train_epoch(999)

    trainNetwork.train(numEnvsTrain=45000,
                       numEnvsTest=5000,
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

from src.mpnet import MPnetBase
import torch
import numpy as np
import os.path as osp
import csv
import sys
from tqdm import tqdm
import pandas as pd
from torch.nn.utils import clip_grad_value_

from torch.utils.data import DataLoader
from MPnet.tools.data_loader_dubins import DubinsDataset

get_numpy = lambda x: x.data.cpu().numpy()


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    borrowed from - https://www.jeremyjordan.me/nn-learning-rate/
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''

    def schedule(epoch):
        return initial_lr * (decay_factor**np.floor(epoch / step_size))

    return schedule


def cyclic_schedule(maximum_lr=1e-3, minimum_lr=1e-4, step_size=100):
    """
    A function that returns the learning rate at a schedule following a cyclic pattern
    """

    def schedule(epoch):
        return (maximum_lr - minimum_lr) * np.cos(
            (epoch % step_size) * np.pi / (2 * step_size)) + minimum_lr

    return schedule


class MPnetTrain(MPnetBase):
    """
    Sets up the training of MPnet
    """

    def __init__(self,
                 load_dataset,
                 n_epochs=1000,
                 batchSize=256,
                 learning_rate=1e-2,
                 opt=None,
                 **kwargs):
        """
        Initialize the MPnet trainer
        """
        super().__init__(**kwargs)
        self.train_loss = []
        self.test_loss = []
        self.start_epoch = 0
        self.n_epochs = n_epochs
        self.batchSize = batchSize
        self.load_dataset = load_dataset
        if opt is None:
            opt = torch.optim.Adagrad
        self.mpNet.set_opt(opt, lr=learning_rate)
        # self.mpNet.set_opt(torch.optim.SGD, lr=learning_rate)

    def set_model_train_epoch(self, epoch):
        fileLoc = osp.join(self.modelPath, 'mpnet_epoch_{}.pkl'.format(epoch))
        self.load_network_parameters(fileLoc)
        data = pd.read_csv(osp.join(self.modelPath, 'progress.csv'))
        self.test_loss = list(data['test_loss'][:epoch])
        self.train_loss = list(data['train_loss'][:epoch])
        self.start_epoch = epoch + 1

    def train(self, numEnvsTrain, numEnvsTest, numPaths, trainDataPath,
              testDataPath):
        """
        A method to train the network with given data
        """
        print('Loading data...')
        # obs, inputs, targets = self.load_dataset(N=numEnvsTrain,
        #                                          NP=numPaths,
        #                                          folder_loc=trainDataPath)
        # trainObs, trainInput, trainTarget = self.format_data(
        #     obs, inputs, targets)

        # obs_test, inputs_test, targets_test = self.load_dataset(
        #     N=numEnvsTest, NP=1, folder_loc=testDataPath)
        # testObs, testInput, testTarget = self.format_data(
        #     obs_test, inputs_test, targets_test)

        train_ds = DubinsDataset(trainDataPath, numEnvsTrain*numPaths)
        train_dl = DataLoader(train_ds, shuffle=True, num_workers = 5, batch_size = self.batchSize)

        test_ds = DubinsDataset(testDataPath, numEnvsTest*numPaths)
        testObs, testInput, testTarget = test_ds[:int(numEnvsTest*numPaths/2)]
        testObs, testInput, testTarget = self.format_data(
            testObs, testInput, testTarget)

        # Setting the learning rate scheduler
        # scheduler = step_decay_schedule(initial_lr=3e-4,
        #                                 decay_factor=0.75,
        #                                 step_size=100)
        scheduler = cyclic_schedule(maximum_lr=3e-4,
                                    minimum_lr=3e-5,
                                    step_size=100)

        # lr_range = np.linspace(-6, 0.1, self.n_epochs / 2)

        # Train the Models
        print('Training...')
        # TODO: Generate indices after knowing the samples generated
        indices = np.arange(numEnvsTrain * numPaths)

        for epoch in range(self.start_epoch, self.n_epochs):
            batch_loss = 0
            # if epoch % 10 == 0:
            #     newLR = scheduler(epoch)
            # newLR = scheduler(epoch)
            # if epoch % 2 == 0:
            #     self.mpNet.set_opt(torch.optim.SGD, 10**lr_range[epoch // 2])
            # np.random.shuffle(indices)
            grad_norm = []
            self.mpNet.train()
            # for i in range((numEnvsTrain * numPaths) // self.batchSize):
            for batch in train_dl:
                bobs, bi, bt = batch
                bobs, bi, bt = self.format_data(bobs, bi, bt)
                # Run gradient descent
                self.mpNet.fit(bobs, bi, bt)
                # grad_norm.append(self.mpNet(bobs, bi, bt))

            with torch.no_grad():
                # self.mpNet.eval()
                network_output = self.mpNet(bi, bobs)
                # Train loss
                train_loss_i = self.mpNet.loss(
                    network_output,
                    bt
                ).sum(dim=1).mean()
                train_loss_i = get_numpy(train_loss_i)

                # Test loss
                network_output = self.mpNet(testInput, testObs)
                test_loss_i = self.mpNet.loss(
                    network_output,
                    testTarget
                    ).sum(dim=1).mean()
                test_loss_i = get_numpy(test_loss_i)

                if train_loss_i > 10:
                    import pdb;pdb.set_trace()

            # print('Epoch {} - mean grad norm {}'.format(epoch, np.mean(grad_norm)))
            print('Epoch {} - train loss: {}'.format(epoch, train_loss_i))
            print('Epoch {} - test loss: {}'.format(epoch, test_loss_i))

            self.train_loss.append(train_loss_i)
            self.test_loss.append(test_loss_i)
            # Save the models
            if (epoch + 1) % 50 == 0:
                model_file = 'mpnet_epoch_%d.pkl' % (epoch)
                self.save_network_state(osp.join(self.modelPath, model_file))

            results = {
                'test_loss': self.test_loss,
                'train_loss': self.train_loss
            }
            # Record training and testing loss
            with open(osp.join(self.modelPath, 'progress.csv'), 'w') as f:
                fieldnames = results.keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                row_data = [
                    dict(
                        zip(fieldnames,
                            [results[key][i] for key in fieldnames]))
                    for i in range(len(results['train_loss']))
                ]
                for row in row_data:
                    writer.writerow(row)

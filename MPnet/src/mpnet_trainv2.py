from src.mpnet import MPnetBase
import torch
import numpy as np
import os.path as osp
import csv
import sys
get_numpy = lambda x: x.data.cpu().numpy()


class MPnetTrain(MPnetBase):
    """
    Sets up the training of MPnet
    """

    def __init__(self, load_dataset, n_epochs=1000, batchSize=256, **kwargs):
        """
        Initialize the MPnet trainer
        """
        super().__init__(**kwargs)

        self.n_epochs = n_epochs
        self.batchSize = batchSize
        self.load_dataset = load_dataset

    def train(self, numEnvs, numPaths, trainDataPath, testDataPath):
        """
        A method to train the network with given data
        """
        obs, inputs, targets = self.load_dataset(N=numEnvs,
                                                 NP=numPaths,
                                                 folder_loc=trainDataPath)
        trainObs, trainInput, trainTarget = self.format_data(
            obs, inputs, targets)

        obs_test, inputs_test, targets_test = self.load_dataset(
            N=50, NP=1, folder_loc=testDataPath)
        testObs, testInput, testTarget = self.format_data(
            obs_test, inputs_test, targets_test)

        # Train the Models
        print('Training...')
        train_loss = []
        test_loss = []
        indices = np.arange(numEnvs * numPaths)

        for epoch in range(self.n_epochs):
            batch_loss = 0
            np.random.shuffle(indices)
            for i in range((numEnvs * numPaths) // self.batchSize):
                sample_index = indices[i * self.batchSize:(i + 1) *
                                       self.batchSize]
                bobs, bi, bt = trainObs[sample_index, ...], trainInput[
                    sample_index, :], trainTarget[sample_index, :]
                # Run gradient descent
                self.mpNet.fit(bobs, bi, bt)

            with torch.no_grad():
                # TODO : need to break this up into smaller chunks
                train_loss_i = get_numpy(
                    self.mpNet.loss(
                        self.mpNet(trainInput[:5000, ...],
                                   trainObs[:5000, ...]),
                        trainTarget[:5000, ...]))
                # Test loss
                test_loss_i = get_numpy(
                    self.mpNet.loss(self.mpNet(testInput, testObs),
                                    testTarget))

            print('Epoch {} - train loss: {}'.format(epoch, train_loss_i))
            print('Epoch {} - test loss: {}'.format(epoch, test_loss_i))
            train_loss.append(train_loss_i)
            test_loss.append(test_loss_i)
            # Save the models
            if epoch % 10 == 0:
                model_file = 'mpnet_epoch_%d.pkl' % (epoch)
                self.save_network_state(osp.join(self.modelPath, model_file))

            results = {'test_loss': test_loss, 'train_loss': train_loss}
            # Record training and testing loss
            with open(osp.join(self.modelPath, 'progress.csv'), 'w') as f:
                fieldnames = results.keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i, j in zip(test_loss, train_loss):
                    writer.writerow({'test_loss': i, 'train_loss': j})

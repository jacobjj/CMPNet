from src.mpnet import MPnetBase
import numpy as np
import os.path as osp
import csv

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
        # NOTE: This is written for a certain format of dataset
        obs, inputs, targets = self.load_dataset(N=numEnvs * numPaths,
                                                 folder_loc=trainDataPath)
        trainObs, trainInput, trainTarget = self.format_data(
            obs, inputs, targets)

        # NOTE: Change the number of test cases for the path
        obs_test, inputs_test, targets_test = self.load_dataset(
            N=4000, folder_loc=testDataPath)
        testObs, testInput, testTarget = self.format_data(
            obs_test, inputs_test, targets_test)

        # Train the Models
        print('Training...')
        train_loss = []
        test_loss = []
        for epoch in range(self.n_epochs):
            data_all = []
            num_path_trained = 0
            print('Epoch : {}'.format(epoch))
            indices = np.arange(numEnvs * numPaths)
            np.random.shuffle(indices)
            batch_loss = 0
            for i in range((numPaths * numEnvs) // self.batchSize):
                sample_index = indices[i * self.batchSize:(i + 1) *
                                       self.batchSize]
                bobs, bi, bt = trainObs[sample_index, ...], trainInput[
                    sample_index, :], trainTarget[sample_index, :]
                # Run gradient descent
                self.mpNet.zero_grad()
                self.mpNet.observe(bobs, bi, 0, bt, False)
                num_path_trained += 1
            loss = get_numpy(
                self.mpNet.loss(self.mpNet(trainObs, trainInput), trainTarget))
            print('Epoch {} - train loss: {}'.format(epoch, loss))
            train_loss.append(loss)
            loss = get_numpy(
                self.mpNet.loss(self.mpNet(testObs, testInput), testTarget))
            print('Epoch {} - test loss: {}'.format(epoch, loss))
            test_loss.append(loss)
            # Save the models
            if epoch > 0:
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

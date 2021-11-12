import torch
import numpy as np
from torch.utils.data import Dataset

class RRCDataset(Dataset):
    """
    Dataset of Robot Remote Control Data.
    Either the testing data set is used or a generalization data set
    """

    def __init__(self, path, num_data, control, train, output_factor, index_offset=0):
        super(RRCDataset, self).__init__()
        self.num_data = num_data
        self.data_len = 50
        self.action_dim = 2
        self.state_dim = 4
        self.output_dim = 4
        self.info_dim = 9
        self.index_offset = index_offset

        self.output_factor = output_factor

        control_string = 'Control'
        if not control:
            control_string = 'NoControl'
        train_string = 'Train'
        if not train:
            train_string = 'Test'

        self.filename = path + control_string +'Data' + train_string + '.npy'
        data_train1 = np.load(self.filename)
        self.input_data, self.output_data, self.info_data = self.__data_to_input_output(data_train1)

    def __data_to_input_output(self, data1):
        input_data1 = np.zeros((self.data_len - 1, self.num_data, self.action_dim + self.state_dim), dtype='float64')
        output_data1 = np.zeros((self.data_len - 1, self.num_data, self.output_dim), dtype='float64')
        info_data1 = np.zeros((self.data_len - 1, self.num_data, self.info_dim), dtype='float64')
        for i in range(self.num_data):
            # Move actions front
            input_data1[:, i, 0:self.action_dim] = data1[i + self.index_offset, 0:(self.data_len - 1),
                                                   self.state_dim:(self.state_dim + self.action_dim)]
            input_data1[:, i, self.action_dim:(self.action_dim + self.state_dim)] = data1[i + self.index_offset,
                                                                                    0:(self.data_len - 1),
                                                                                    0:self.state_dim]
            info_data1[:, i, 0:self.info_dim] = data1[i + self.index_offset, 0:(self.data_len - 1),
                                                (self.state_dim + self.action_dim):(
                                                        (self.state_dim + self.action_dim) + self.info_dim)]
            output_data1[:, i, 0:self.output_dim] = 1.0 / self.output_factor * (
                    data1[i + self.index_offset, 1:self.data_len, 0:self.output_dim] - data1[i + self.index_offset,
                                                                                 0:(self.data_len - 1),
                                                                                 0:self.output_dim])

        return torch.from_numpy(input_data1).float(), torch.from_numpy(output_data1).float(), torch.from_numpy(
            info_data1).float()

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return self.input_data[:, idx, :], self.output_data[:, idx, :], self.info_data[:, idx, :]
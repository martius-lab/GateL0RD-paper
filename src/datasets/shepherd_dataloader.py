import torch
import numpy as np
from torch.utils.data import Dataset

class ShepherdDataset(Dataset):

    def __init__(self, data_path, output_factor=0.1):
        super(ShepherdDataset, self).__init__()
        self.num_data = 8000
        self.data_len = 100
        self.action_dim = 3
        self.state_dim = 7
        self.output_dim = 7
        self.info_dim = 11
        self.output_factor = output_factor

        self.filename = data_path
        data_train1 = np.load(self.filename)
        self.input_data, self.output_data, self.info_data = self.__data_to_input_output(data_train1)

    def __data_to_input_output(self, data1):
        input_data1 = np.zeros((self.data_len, self.num_data, self.action_dim + self.state_dim), dtype='float64')
        output_data1 = np.zeros((self.data_len, self.num_data, self.output_dim), dtype='float64')
        info_data1 = np.zeros((self.data_len, self.num_data, self.info_dim), dtype='float64')
        for i in range(self.num_data):
            # Move actions front
            input_data1[:, i, 0:self.action_dim] = data1[i, 0:self.data_len,
                                                   self.state_dim:(self.state_dim + self.action_dim)]
            input_data1[:, i, self.action_dim:(self.action_dim + self.state_dim)] = data1[i, 0:self.data_len,
                                                                                    0:self.state_dim]

            info_data1[:, i, 0:self.info_dim] = data1[i, 0:self.data_len, (self.state_dim + self.action_dim):(
                        (self.state_dim + self.action_dim) + self.info_dim)]

            output_data1[:, i, 0:self.output_dim] = 1.0 / self.output_factor * (data1[i, 1:(self.data_len + 1), 0:self.output_dim]
                                                                                - data1[i, 0:self.data_len, 0:self.output_dim])

        return torch.from_numpy(input_data1).float(), torch.from_numpy(output_data1).float(), torch.from_numpy(info_data1).float()

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return self.input_data[:, idx, :], self.output_data[:, idx, :], self.info_data[:, idx, :]


def get_shepherd_dataloaders(dataset_split_rs, subdataset_batch_size, factor_output,
                             num_data_train, num_data_val, num_data_test):

    data_path = 'data/Shepherd/'
    suffix = ['_gateclosed_startleft.npy', '_gateopen_startleft.npy', '_gateopen_startright.npy',
              '_sheepcaught_startright.npy']

    dataloaders_train = []
    dataloaders_val = []
    dataloaders_test = []

    for i in range(4):
        dataset = ShepherdDataset(data_path=data_path + 'shepherd_data' + suffix[i], output_factor=factor_output)

        dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(dataset, [num_data_train, num_data_val,
                                                                                           num_data_test],
                                                                                 generator=torch.Generator().manual_seed(
                                                                                     dataset_split_rs))
        # Training
        dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=subdataset_batch_size, shuffle=True)
        dataloaders_train.append(dataloader)

        # Validation:
        val_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=subdataset_batch_size, shuffle=True)
        dataloaders_val.append(val_dataloader)

        # Testing:
        test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=subdataset_batch_size, shuffle=True)
        dataloaders_test.append(test_dataloader)
    return dataloaders_train, dataloaders_val, dataloaders_test
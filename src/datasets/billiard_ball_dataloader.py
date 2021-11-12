import torch
import numpy as np
from torch.utils.data import Dataset

class BilliardDataset(Dataset):

    def __init__(self, path, name, seq_len, start_time):
        self.train_len = seq_len
        self.state_dim = 2
        self.output_dim = 2
        self.filename = path
        self.start_time = start_time

        data_inp = np.load(self.filename + name + '.npy')
        input_data1 = np.moveaxis(data_inp, 0, 1)
        self.input_data = torch.from_numpy(input_data1).float()
        self.data_len, self.num_data, _ = self.input_data.shape

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        start_time = self.start_time
        inp1 = self.input_data[start_time:(start_time + self.train_len), idx,:]
        out1 = self.input_data[(start_time + 1):(start_time + self.train_len + 1), idx, :]
        return inp1, out1


def get_billiard_ball_dataloaders(dataset_split_rs, seq_len, subdataset_batch_size):
    '''
    Loading the data:
    We have 6 data sets composed of 1200 sequences of 150 time steps. In 3 of the data sets the ball drops into the
    pocket (datasets named pocket_drop_X with X in {50, 100, 150}; [X-50, X] marks the interval in which the ball ends
    up in the pocket). We want to balance out for each batch how often pocket drops occur and at what velocity (~ point in time).
    For that we split the datasets into 16 datasets. In 1/4 the ball ends up in the pocket (3/16 with pocket drops
    occurring during the sequence; 1/16 ball is already within pocket), in 3/4 the ball is on the table over the course
    of the whole sequence. Additionally, we balance out the time whithin the 150 sequence to get multiple velocities within
    each batch.
    '''

    subdataset_size_train = 800
    subdataset_size_test = 200
    dataset_path = 'data/BilliardBall/'
    dataset_names = ['pocket_drop_100', 'pocket_drop_150', 'no_pocket_drop_1', 'no_pocket_drop_2', 'no_pocket_drop_3']
    start_times = [0, 50, 100]

    dataset1 = BilliardDataset(path=dataset_path, name='pocket_drop_50', seq_len=seq_len, start_time=0)
    train_data1, val_data1, test_data1 = torch.utils.data.random_split(dataset1, [subdataset_size_train,
                                                                                  subdataset_size_test,
                                                                                  subdataset_size_test], generator=torch.Generator().manual_seed(dataset_split_rs))
    train_dataloader1 = torch.utils.data.DataLoader(train_data1, batch_size=subdataset_batch_size, shuffle=True)
    val_dataloader1 = torch.utils.data.DataLoader(val_data1, batch_size=subdataset_batch_size, shuffle=True)
    test_dataloader1 = torch.utils.data.DataLoader(test_data1, batch_size=subdataset_batch_size, shuffle=True)

    train_dataloaders = [train_dataloader1]
    val_dataloaders = [val_dataloader1]
    test_dataloaders = [test_dataloader1]

    for data_name in dataset_names:

        for s_t in start_times:
            dataset_i = BilliardDataset(path=dataset_path, name=data_name, seq_len=seq_len, start_time=s_t)
            train_data_i, val_data_i, test_data_i = torch.utils.data.random_split(dataset_i,
                                                                                  [subdataset_size_train,
                                                                                   subdataset_size_test,
                                                                                   subdataset_size_test],
                                                                                  generator=torch.Generator().manual_seed(
                                                                                      dataset_split_rs))

            train_dataloader_i = torch.utils.data.DataLoader(train_data_i, batch_size=subdataset_batch_size,
                                                             shuffle=True)
            val_dataloader_i = torch.utils.data.DataLoader(val_data_i, batch_size=subdataset_batch_size, shuffle=True)
            test_dataloader_i = torch.utils.data.DataLoader(test_data_i, batch_size=subdataset_batch_size, shuffle=True)

            train_dataloaders.append(train_dataloader_i)
            val_dataloaders.append(val_dataloader_i)
            test_dataloaders.append(test_dataloader_i)
    return train_dataloaders, val_dataloaders, test_dataloaders
import torch
import numpy as np
from torch.utils.data import Dataset

class FPPGraspDataset(Dataset):
    """
    Dataset of Fetch Pick & Place. Filtered to only include reach-, grasp- & carry-sequences
    """

    def __init__(self, path, name, relevant_dims, seq_len):
        super(FPPGraspDataset, self).__init__()


        self.seq_len = seq_len

        action_data = np.load(path + 'act_' + name + '.npy')
        act_shape = action_data.shape

        assert act_shape[1] == 50
        assert act_shape[2] == 4

        observation_data = np.load(path + 'obs_' + name + '.npy')
        next_observation_data = np.load(path + 'next_obs_' + name + '.npy')

        self.num_data = observation_data.shape[0]
        assert act_shape[0] == self.num_data

        self.act_data = action_data[:, 0:self.seq_len, :]
        self.obs_data = observation_data[:, 0:self.seq_len, relevant_dims]
        self.next_obs_data = next_observation_data[:, 0:self.seq_len, relevant_dims]


    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return self.obs_data[idx, :, :], self.act_data[idx, :, :], (self.next_obs_data[idx, :, :] - self.obs_data[idx, :, :])


def create_all_dataloaders(relevant_dims, seq_len, dataset_split_rs, num_data_train, num_data_test, num_data_gen, batch_size):


    path = 'data/FetchPickAndPlace/'
    dataset = FPPGraspDataset(path=path, name='grasp_at_5', relevant_dims=relevant_dims, seq_len=seq_len)

    num_data_ignore = len(dataset) - 2*num_data_test - num_data_train
    train_dataset, val_dataset, test_dataset, _ = torch.utils.data.random_split(dataset,
                                                                                [num_data_train, num_data_test, num_data_test, num_data_ignore],
                                                                                generator=torch.Generator().manual_seed(dataset_split_rs))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Generalization dataloader
    gen_dataset = FPPGraspDataset(path=path, name='grasp_at_later', relevant_dims=relevant_dims, seq_len=seq_len)
    gen_num_data_ignore = len(gen_dataset) - num_data_gen
    gen_test_dataset, _ = torch.utils.data.random_split(gen_dataset, [num_data_gen, gen_num_data_ignore],
                                                        generator=torch.Generator().manual_seed(dataset_split_rs))
    gen_dataloader = torch.utils.data.DataLoader(gen_test_dataset, batch_size=batch_size, shuffle=True)


    return train_dataloader, val_dataloader, test_dataloader, gen_dataloader



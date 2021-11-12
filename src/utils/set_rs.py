import torch
import numpy as np
import random

def set_rs(rand_seed):
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
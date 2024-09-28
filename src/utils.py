import numpy as np
import torch
import pickle


def sample_batch(batch_size, dataset, device):
    sample = torch.from_numpy(dataset[np.random.choice(
        dataset.shape[0], batch_size, replace=False)]).float().to(device)
    return sample


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
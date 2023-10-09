from MMapDataset import MMapDataset
from torch.utils.data import DataLoader, random_split
from torch import Generator
import numpy as np
import torch

def split_data(seqs, labels, train_split, val_split, test_split, fasta, length, use_annotations=False):
    data = MMapDataset(seqs, labels, fasta, length, use_annotations)
    training_data, val_data, test_data = random_split(
        data, [train_split, val_split, test_split], generator=Generator().manual_seed(42))
    return training_data, val_data, test_data

def split_kfold(data):
    folds = random_split(data, [0.2, 0.2, 0.2, 0.2, 0.2], generator=Generator().manual_seed(42))
    return folds

def get_data_loader(data, batch_size, sampler=None):
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32,
        persistent_workers=True,
        pin_memory=True,
        sampler=sampler,
    )
    return loader

"""Custom Dataset class for our data"""

import glob
import os
import zipfile

import h5py
import lava.lib.dl.slayer as slayer
import numpy as np
# import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset


class TrafficDataset(Dataset):
    """Traffic dataset method

    Parameters
    ----------
    """

    def __init__(
        self,
        path: str,
        train: bool = True,
        transform=None,
        sampling_time: int = 1,
        sample_length: int = 1050,
    ) -> None:
        super(TrafficDataset, self).__init__()
        self.path: str = path
        if train:
            data_path = os.path.join(path, "Train")
        else:
            data_path = os.path.join(path, "Test")

        self.samples = glob.glob(f"{data_path}/*/*.bin")
        self.sampling_time = sampling_time
        self.num_time_bins = int(sample_length / sampling_time)
        self.transform = transform
        self.labels = {"light": 0, "medium": 1, "heavy": 2}

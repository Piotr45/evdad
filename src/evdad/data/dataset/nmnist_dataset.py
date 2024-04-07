import glob
import os

import lava.lib.dl.slayer as slayer
import numpy as np
import torch
from torch.utils.data import Dataset

from evdad.data.utils.augment import augment_events


class NMNISTDataset(Dataset):
    """NMNIST dataset class."""

    def __init__(
        self,
        data_path: str,
        img_shape: tuple,
        sampling_time: int,
        sample_length: int,
        num_classes: int,
        reshape_spike: bool,
        data_is_label: bool = False,
    ):
        super(NMNISTDataset, self).__init__()

        self.data_csv: str = data_path

        self.img_shape: tuple = img_shape
        self.sampling_time: int = sampling_time
        self.sample_length: int = sample_length
        self.num_time_bins: int = int(sample_length / sampling_time)

        self.reshape_spike: bool = reshape_spike
        self.data_is_label: bool = data_is_label

        self.num_classes: int = num_classes

        self._data: list = glob.glob(f"{data_path}/*/*.bin")

    def __getitem__(self, index: int):
        filename = self._data[index]

        event = slayer.io.read_2d_spikes(filename)

        event = augment_events(event)

        spike = event.fill_tensor(
            torch.zeros(2, self.img_shape[0], self.img_shape[1], self.num_time_bins),
            sampling_time=self.sampling_time,
        )

        if self.data_is_label:
            label = spike.detach().clone()
        else:
            label = os.path.basename(os.path.abspath(os.path.join(filename, os.pardir)))

        if self.reshape_spike:
            return (
                spike.reshape(-1, self.num_time_bins),
                label.reshape(-1, self.num_time_bins) if self.data_is_label else label,
            )
        return spike, label

    def __len__(self):
        return len(self._data)

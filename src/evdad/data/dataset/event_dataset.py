"""Event-driven Dataset class"""

import os

import lava.lib.dl.slayer as slayer
import numpy as np
import torch
from torch.utils.data import Dataset

from evdad.data.utils import read_csv, read_label_csv


class EventDataset(Dataset):
    """"""

    def __init__(
        self,
        data_csv: str,
        labels_csv: str,
        img_shape: tuple,
        sampling_time: int,
        sample_length: int,
        num_classes: int,
        reshape_spike: bool,
    ) -> None:
        self.data_csv: str = data_csv
        self.labels_csv: str = labels_csv

        self.img_shape: tuple = img_shape
        self.sampling_time: int = sampling_time
        self.sample_length: int = sample_length
        self.num_time_bins: int = int(sample_length / sampling_time)

        self.reshape_spike: bool = reshape_spike

        self.num_classes: int = num_classes
        # self.labels: dict = {"light": 0, "medium": 1, "heavy": 2}

        self._data: list = read_csv(data_csv)
        self._labels: list = read_csv(labels_csv)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple:
        """Gets data and label from dataset"""
        labels = self._read_label(self._labels[index])
        # event = slayer.io.read_2d_spikes(self._data[index])
        # spike = event.to_tensor()#[:, :, :, int(self._data[index][1]):int(self._data[index][2])]

        spike = np.load(self._data[index])
        spike = torch.from_numpy(spike)
        # print(spike.shape, self._data[index])
        # zeros = torch.zeros((2, 256, 256, 1))
        # zeros[: spike.shape[0], : spike.shape[1], : spike.shape[2], : spike.shape[3]] = spike
        # zeros = zeros[:, :, :, 0]

        if self.reshape_spike:
            return spike.reshape(-1, 1), labels
        #     return zeros.reshape(-1, spike.shape[-1]), labels
        return spike, labels

    @staticmethod
    def _read_label(path: str) -> str:
        """Funtion that reads label file.

        Args:
            path: Path to label file

        Returns:
            Label in the desired form
        """
        # TODO change it in the future
        # return read_label_csv(path)[0]
        return 0
        # return np.load(path)

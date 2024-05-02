"""Event-driven Dataset class"""

import os

import lava.lib.dl.slayer as slayer
import numpy as np
import torch
from torch.utils.data import Dataset

from evdad.data.utils.io import read_csv, read_label_csv


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
        data_is_label: bool = False,
    ) -> None:
        self.data_csv: str = data_csv
        self.labels_csv: str = labels_csv

        self.img_shape: tuple = img_shape
        self.sampling_time: int = sampling_time
        self.sample_length: int = sample_length
        self.num_time_bins: int = int(sample_length / sampling_time)

        self.reshape_spike: bool = reshape_spike
        self.data_is_label: bool = data_is_label

        self.num_classes: int = num_classes
        self.labels: dict = {"light": 0, "medium": 1, "heavy": 2}

        self._data: list = read_csv(data_csv)
        self._labels: list = read_csv(labels_csv)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple:
        """Gets data and label from dataset"""
        labels = self._read_label(self._labels[index])

        spike = np.load(self._data[index]).astype(np.float32)
        spike = torch.from_numpy(spike)

        if self.data_is_label:
            label = spike.detach().clone()
        else:
            # label = labels
            label = self.labels[labels]

        if self.reshape_spike:
            return (
                spike.reshape(-1, self.num_time_bins),
                label.reshape(-1, self.num_time_bins) if self.data_is_label else label,
            )
        return spike, label

    # @staticmethod
    def _read_label(self, path: str) -> str:
        """Funtion that reads label file.

        Args:
            path: Path to label file

        Returns:
            Label in the desired form
        """
        # TODO change it in the future
        return read_label_csv(path)[0]
        return torch.tensor([self.labels[label] for label in read_label_csv(path)], dtype=torch.long)

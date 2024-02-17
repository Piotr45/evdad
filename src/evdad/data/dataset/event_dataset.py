"""Event-driven Dataset class"""

import lava.lib.dl.slayer as slayer
import torch
from torch.utils.data import Dataset

from evdad.data.utils import read_csv


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
    ) -> None:
        self.data_csv: str = data_csv
        self.labels_csv: str = labels_csv

        self.img_shape: tuple = img_shape
        self.sampling_time: int = sampling_time
        self.sample_length: int = sample_length
        self.num_time_bins: int = int(sample_length / sampling_time)

        self.num_classes: int = num_classes
        self.labels: dict = {"light": 0, "medium": 1, "heavy": 2}

        self._data: list = read_csv(data_csv)
        self._labels: list = read_csv(labels_csv)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple:
        """Gets data and label from dataset"""
        label = self.read_label(self._labels[index])
        event = slayer.io.read_2d_spikes(self._data[index])
        spike = event.to_tensor()
        spike = event.fill_tensor(
            torch.zeros(2, self.img_shape[0], self.img_shape[1], self.num_time_bins),
            sampling_time=self.sampling_time,
        )
        return spike.reshape(-1, self.num_time_bins), self.labels[label]

    @staticmethod
    def read_label(path: str) -> str:
        """Funtion that reads label file.

        Args:
            path: Path to label file

        Returns:
            Label in the desired form
        """
        # TODO change it in the future
        return read_csv(path)[0]

"""Event-driven Dataset class"""

import math
import os
from typing import Optional

import lava.lib.dl.slayer as slayer
import numpy as np
import torch
from lava.lib.dl.slayer.io import read_2d_spikes
from torch.utils.data import Dataset

from evdad.data.utils.io import parse_dataset_input, read_csv
from evdad.types import PathStr


class EventDataset(Dataset):
    """Video Event dataset class."""

    def __init__(
        self,
        data: PathStr,
        labels: Optional[PathStr],
        ST: int,
        SL: int,
        OL: int,
        C: int,
        H: int,
        W: int,
        use_cache: bool = True,
    ) -> None:
        """Video Event dataset class.

        Args:
            data: Path to csv file with input data
            labels: Path to csv file with event labels
            ST: Sampling time
            SL: Sample length
            OL: Overlap length
            C: Number of channels
            H: Height of image
            W: Width of image
            use_cache: Whether to use cached data
        """
        self.events: dict[str, PathStr] = {
            os.path.splitext(os.path.basename(path))[0]: path for path in parse_dataset_input(data, "bin")
        }
        if labels is not None:
            self.labels: dict[str, PathStr] = {
                os.path.splitext(os.path.basename(path))[0]: path for path in parse_dataset_input(labels, "csv")
            }
        else:
            self.labels = None

        self.sampling_time: int = ST
        self.sample_length: int = SL
        self.overlap_length: int = OL
        self.num_channels: int = C
        self.height: int = H
        self.width: int = W

        self.num_time_bins: int = int(self.sample_length / self.sampling_time)

        self.metadata: list[tuple[str, int, int, int]] = self.create_metadata()

        self.use_cache: bool = use_cache
        self.cache: dict = self._create_cache()

    def create_metadata(self) -> list[tuple[str, int, int, int]]:
        metadata = []
        for filename, events_path in self.events.items():
            # Read number of frames based on nuber of labels
            spikes = read_2d_spikes(events_path).to_tensor(self.sampling_time)
            num_frames = spikes.shape[-1]
            num_windows = math.floor(num_frames / self.num_time_bins)
            for i in range(num_windows):
                # Path, frame begin, frame end
                metadata.append((filename, i * self.num_time_bins, (i + 1) * self.num_time_bins, num_frames))
        return metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple:
        """Gets data and label from dataset."""
        spikes = self.get_item_data(index)
        if self.labels is not None:
            labels = self.get_item_label(index)
            return spikes, labels
        return spikes, spikes.clone()

    def get_item_data(self, index: int) -> torch.Tensor:
        filename, begin, end, num_frames = self.metadata[index]

        if self.use_cache and filename in self.cache["spikes"].keys():
            # Use cached data
            spikes = self.cache["spikes"][filename]
        else:
            event = read_2d_spikes(self.events[filename])
            spikes = torch.zeros((self.num_channels, self.height, self.width, num_frames), dtype=torch.float32)
            spikes = event.fill_tensor(spikes, sampling_time=self.sampling_time)
            if self.use_cache:
                self.cache["spikes"][filename] = spikes
        return spikes[..., begin:end]

    def get_item_label(self, index: int) -> torch.Tensor:
        filename, begin, end, num_frames = self.metadata[index]

        if self.use_cache and filename in self.cache["labels"].keys():
            # Use cached data
            labels = self.cache["labels"][filename]
        else:
            labels = self._read_label(self.labels[filename])
            if self.use_cache:
                self.cache["labels"][filename] = labels
        return labels[begin:end][0]

    @staticmethod
    def _read_label(path: str) -> torch.Tensor:
        """Function that reads label file.

        Args:
            path: Path to label file

        Returns:
            Label in the desired form
        """
        return torch.tensor([int(label) for label in read_csv(path)], dtype=torch.long)

    def _create_cache(self) -> dict:
        if self.labels is not None:
            return {"spikes": {}, "labels": {}}
        return {"spikes": {}}

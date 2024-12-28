import numpy as np
from torch.utils.data import DataLoader
from utils import get_test_data_dir

from evdad.data.dataset.event_dataset import EventDataset
from evdad.data.dataset.generic_event_dataset import GenericEventDataset


def test_generic_event_dataset():
    train_data_path = f"{get_test_data_dir()}/dataset/train.csv"
    train_labels_path = f"{get_test_data_dir()}/dataset/train_labels.csv"
    ST, SL, OL, C, H, W = 1, 30, 5, 2, 128, 128

    generic_event_dataset = GenericEventDataset(
        train_data_path, train_labels_path, train_data_path, train_labels_path, ST, SL, OL, C, H, W
    )

    train_dataset = generic_event_dataset.get_train_dataset()
    test_dataset = generic_event_dataset.get_test_dataset()

    assert type(train_dataset) == EventDataset
    assert type(test_dataset) == EventDataset

    assert train_dataset.sampling_time == ST
    assert train_dataset.sample_length == SL
    assert train_dataset.overlap_length == OL
    assert train_dataset.num_channels == C
    assert train_dataset.height == H
    assert train_dataset.width == W
    assert len(train_dataset) == 15

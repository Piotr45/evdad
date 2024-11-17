import numpy as np
from torch.utils.data import DataLoader
from utils import get_test_data_dir

from evdad.data.dataset.event_dataset import EventDataset


def test_event_dataset():
    ST, SL, OL, C, H, W = 1, 30, 5, 2, 128, 128
    dataset = EventDataset(
        f"{get_test_data_dir()}/dataset/train.csv",
        f"{get_test_data_dir()}/dataset/train_labels.csv",
        ST,
        SL,
        OL,
        C,
        H,
        W,
    )

    assert dataset.sampling_time == ST
    assert dataset.sample_length == SL
    assert dataset.overlap_length == OL
    assert dataset.num_channels == C
    assert dataset.height == H
    assert dataset.width == W
    assert len(dataset) == 15


def test_event_dataset_with_dirs():
    ST, SL, OL, C, H, W = 1, 30, 5, 2, 128, 128
    dataset = EventDataset(
        f"{get_test_data_dir()}/dataset/data",
        f"{get_test_data_dir()}/dataset/labels",
        ST,
        SL,
        OL,
        C,
        H,
        W,
    )

    assert dataset.sampling_time == ST
    assert dataset.sample_length == SL
    assert dataset.overlap_length == OL
    assert dataset.num_channels == C
    assert dataset.height == H
    assert dataset.width == W
    assert len(dataset) == 15


def test_data_event_dataloader():
    ST, SL, OL, C, H, W = 1, 30, 5, 2, 128, 128
    dataset = EventDataset(
        f"{get_test_data_dir()}/dataset/train.csv",
        f"{get_test_data_dir()}/dataset/train_labels.csv",
        ST,
        SL,
        OL,
        C,
        H,
        W,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    events, labels = next(iter(dataloader))

    assert events.shape == (1, C, H, W, SL)
    assert labels.shape == (1,)

    assert (events[0, 0, 5, 51, 5:10].numpy() == np.array([0, 0, 0, 1, 0])).all()

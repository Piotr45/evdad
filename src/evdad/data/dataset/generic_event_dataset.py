"""Event-driven Dataset class"""

from evdad.data.dataset.event_dataset import EventDataset


class GenericEventDataset:
    """Event-driven dataset."""

    def __init__(
        self,
        train_data: str,
        train_labels: str,
        test_data: str,
        test_labels: str,
        img_shape: tuple[int, int],
        sampling_time: int,
        sample_length: int,
        num_classes: int,
        reshape_spike: bool,
    ) -> None:
        # Paths
        self.train_data: str = train_data
        self.train_labels: str = train_labels
        self.test_data: str = test_data
        self.test_labels: str = test_labels

        # Data properties
        self.img_shape: tuple = img_shape
        self.sampling_time: int = sampling_time
        self.sample_length: int = sample_length

        # General info
        self.num_classes: int = num_classes
        self.reshape_spike: bool = reshape_spike

    def get_train_dataset(self) -> EventDataset:
        return EventDataset(
            self.train_data,
            self.train_labels,
            self.img_shape,
            self.sampling_time,
            self.sample_length,
            self.num_classes,
            self.reshape_spike,
        )

    def get_test_dataset(self) -> EventDataset:
        return EventDataset(
            self.test_data,
            self.test_labels,
            self.img_shape,
            self.sampling_time,
            self.sample_length,
            self.num_classes,
            self.reshape_spike,
        )

"""Event-driven Dataset class"""

from evdad.data.dataset.nmnist_dataset import NMNISTDataset


class GenericNMNISTDataset:
    """Event-driven NMNIST dataset."""

    def __init__(
        self,
        train_data: str,
        test_data: str,
        img_shape: tuple[int, int],
        sampling_time: int,
        sample_length: int,
        num_classes: int,
        reshape_spike: bool = False,
        data_is_label: bool = False,
    ) -> None:
        # Paths
        self.train_data: str = train_data
        self.test_data: str = test_data

        # Data properties
        self.img_shape: tuple = img_shape
        self.sampling_time: int = sampling_time
        self.sample_length: int = sample_length

        # General info
        self.num_classes: int = num_classes
        self.reshape_spike: bool = reshape_spike
        self.data_is_label: bool = data_is_label

    def get_train_dataset(self) -> NMNISTDataset:
        return NMNISTDataset(
            self.train_data,
            self.img_shape,
            self.sampling_time,
            self.sample_length,
            self.num_classes,
            self.reshape_spike,
            self.data_is_label,
        )

    def get_test_dataset(self) -> NMNISTDataset:
        return NMNISTDataset(
            self.test_data,
            self.img_shape,
            self.sampling_time,
            self.sample_length,
            self.num_classes,
            self.reshape_spike,
            self.data_is_label,
        )

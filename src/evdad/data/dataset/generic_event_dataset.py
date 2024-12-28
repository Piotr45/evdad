"""Event-driven Dataset class"""

from evdad.data.dataset.event_dataset import EventDataset
from evdad.types import PathStr


class GenericEventDataset:
    """Event-driven dataset."""

    def __init__(
        self,
        train_data: PathStr,
        train_labels: PathStr,
        test_data: PathStr,
        test_labels: PathStr,
        ST: int,
        SL: int,
        OL: int,
        C: int,
        H: int,
        W: int,
        use_cache: bool = False,
    ) -> None:
        # Paths
        self.train_data: PathStr = train_data
        self.train_labels: PathStr = train_labels
        self.test_data: PathStr = test_data
        self.test_labels: PathStr = test_labels

        # Data properties
        self.ST: int = ST
        self.SL: int = SL
        self.OL: int = OL
        self.C: int = C
        self.H: int = H
        self.W: int = W

        # Dataloader properties
        self.use_cache: bool = use_cache

    def get_train_dataset(self) -> EventDataset:
        return EventDataset(
            self.train_data,
            self.train_labels,
            self.ST,
            self.SL,
            self.OL,
            self.C,
            self.H,
            self.W,
            self.use_cache,
        )

    def get_test_dataset(self) -> EventDataset:
        return EventDataset(
            self.test_data,
            self.test_labels,
            self.ST,
            self.SL,
            self.OL,
            self.C,
            self.H,
            self.W,
            self.use_cache,
        )

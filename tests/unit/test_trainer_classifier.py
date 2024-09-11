import torch
from lava.lib.dl.slayer.classifier import Rate

from evdad.trainer.classifier import get_classifier


def test_get_classifier():
    cfg = {"loss": {"loss_function": "SpikeRate"}}

    classifier = get_classifier(cfg)
    assert isinstance(classifier, type(Rate.predict))

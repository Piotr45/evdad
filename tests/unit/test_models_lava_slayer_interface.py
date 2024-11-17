import pytest
import torch

from evdad.models.lava.slayer.interface import EVDADModel


def test_evdad_model_interface():
    model = EVDADModel()
    with pytest.raises(NotImplementedError):
        model.forward(torch.randn(1))

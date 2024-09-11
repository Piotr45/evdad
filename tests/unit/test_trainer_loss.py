import torch
from lava.lib.dl.slayer.loss import SpikeRate, SpikeTime

from evdad.trainer.loss import get_loss_function


def test_get_loss_function():
    cfg_spike_time = {"loss": {"loss_function": "SpikeTime", "time_constant": 100, "length": 30, "filter_order": 2}}
    cfg_spike_rate = {"loss": {"loss_function": "SpikeRate", "true_rate": 0.3, "false_rate": 0.2}}
    device = torch.device("cpu")

    loss_function = get_loss_function(cfg_spike_time, device)
    assert isinstance(loss_function, SpikeTime)
    loss_function = get_loss_function(cfg_spike_rate, device)
    assert isinstance(loss_function, SpikeRate)

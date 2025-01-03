import logging
from typing import Callable, Union

import lava.lib.dl.slayer as slayer
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

REDUCTION = "sum"


def get_loss_function(cfg: dict, device: str) -> Union[torch.nn.Module, Callable]:
    """Get loss function based on given config.

    Args:
        cfg: Hydra config.
        device: Torch device.

    Returns:
        Loss function.
    """
    loss_function = cfg["loss"]["loss_function"]

    if loss_function == "SpikeTime":
        time_constant = cfg["loss"]["time_constant"] if "time_constant" in list(cfg["loss"].keys()) else 5
        length = cfg["loss"]["length"] if "length" in list(cfg["loss"].keys()) else 100
        filter_order = cfg["loss"]["filter_order"] if "filter_order" in list(cfg["loss"].keys()) else 1
        reduction = cfg["loss"]["reduction"] if "reduction" in list(cfg["loss"].keys()) else "sum"
        return slayer.loss.SpikeTime(
            time_constant=time_constant,
            length=length,
            filter_order=filter_order,
            reduction=reduction,
        ).to(device)
    elif loss_function == "SpikeRate":
        true_rate = cfg["loss"]["true_rate"]
        false_rate = cfg["loss"]["false_rate"]
        reduction = cfg["loss"]["reduction"] if "reduction" in list(cfg["loss"].keys()) else "sum"
        moving_window = cfg["loss"]["moving_window"] if "moving_window" in list(cfg["loss"].keys()) else None
        return slayer.loss.SpikeRate(
            true_rate=true_rate,
            false_rate=false_rate,
            moving_window=moving_window,
            reduction=reduction,
        ).to(device)
    elif loss_function == "MSE":
        REDUCTION = cfg["loss"]["reduction"] if "reduction" in list(cfg["loss"].keys()) else "sum"
        return custom_mse
    elif loss_function == "CrossEntropy":
        return custom_cross_entropy
    elif loss_function == "EventRateLoss":
        return event_rate_loss
    else:
        raise NotImplementedError("This function is not implemented or does not exist.")


def custom_mse(output, target, reduction: str = REDUCTION):
    return torch.nn.functional.mse_loss(output, target, reduction=reduction)


def custom_cross_entropy(output, target):
    return torch.nn.functional.cross_entropy(output, target)


def event_rate_loss(x, max_rate=0.01):
    mean_event_rate = torch.mean(torch.abs(x))
    return F.mse_loss(F.relu(mean_event_rate - max_rate), torch.zeros_like(mean_event_rate))

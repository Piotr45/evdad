"""Dataset augmentation module."""
import math

import lava.lib.dl.slayer as slayer
import numpy as np
import torch


def augment_events(
    event: slayer.io.Event, x_shift: int = 4, y_shift: int = 4, theta: int = 10
) -> slayer.io.Event:
    """TODO"""
    xjitter = np.random.randint(2 * x_shift) - x_shift
    yjitter = np.random.randint(2 * y_shift) - y_shift

    ajitter = (np.random.rand() - 0.5) * theta / 180 * math.pi

    sin_theta = np.sin(ajitter)
    cos_theta = np.cos(ajitter)

    event.x = event.x * cos_theta - event.y * sin_theta + xjitter
    event.y = event.x * sin_theta + event.y * cos_theta + yjitter
    return event


def squash_events(spike: torch.Tensor) -> torch.Tensor:
    C, H, W, T = spike.shape
    output_spike = torch.zeros(shape=(C, H, W, 1))

    # for i in range(C):
    #     for j in range(H):
    #         for k in range(W):
    #             for

    return

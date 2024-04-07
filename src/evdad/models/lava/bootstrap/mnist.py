import os

import h5py
import lava.lib.dl.bootstrap as bootstrap
# import slayer from lava-dl
import lava.lib.dl.slayer as slayer
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        neuron_params = {
            "threshold": 1.25,
            "current_decay": 1,  # this must be 1 to use batchnorm
            "voltage_decay": 0.03,
            "tau_grad": 1,
            "scale_grad": 1,
        }
        neuron_params_drop = {
            **neuron_params,
            # 'dropout' : slayer.neuron.Dropout(p=0.05),
            # 'norm'    : slayer.neuron.norm.MeanOnlyBatchNorm,
        }

        self.blocks = torch.nn.ModuleList(
            [
                # enable affine transform at input
                bootstrap.block.cuba.Input(neuron_params, weight=1, bias=0),
                bootstrap.block.cuba.Dense(neuron_params_drop, 28 * 28, 512, weight_norm=True, weight_scale=2),
                bootstrap.block.cuba.Dense(neuron_params_drop, 512, 512, weight_norm=True, weight_scale=2),
                bootstrap.block.cuba.Affine(neuron_params, 512, 10, weight_norm=True, weight_scale=2),
            ]
        )

    def forward(self, x, mode):
        count = []
        N, C, H, W = x.shape
        if mode.base_mode == bootstrap.Mode.ANN:
            x = x.reshape([N, C, H, W, 1])
        else:
            x = slayer.utils.time.replicate(x, 16)

        x = x.reshape(N, -1, x.shape[-1])

        for block, m in zip(self.blocks, mode):
            x = block(x, mode=m)
            count.append(torch.mean(x).item())

        x = torch.mean(x, dim=-1).reshape((N, -1))

        return x, torch.FloatTensor(count).reshape((1, -1)).to(x.device)

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, "synapse")]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + "gradFlow.png")
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, "w")
        layer = h.create_group("layer")
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f"{i}"))

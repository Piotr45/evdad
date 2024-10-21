import h5py
import torch.nn as nn


class EVDADModel(nn.Module):
    def __init__(self):
        super(EVDADModel, self).__init__()

    def forward(self, spike):
        raise NotImplementedError

    def export_hdf5(self, filename: str) -> None:
        # network export to hdf5 format
        h = h5py.File(filename, "w")
        layer = h.create_group("layer")
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f"{i}"))

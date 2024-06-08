import lava.lib.dl.slayer as slayer
import torch


class SimpleConv(torch.nn.Module):
    """Simple Conv model class."""

    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        T: int,
        threshold: float,
        current_decay: float,
        voltage_decay: float,
        tau_grad: float,
        scale_grad: float,
        requires_grad: bool,
    ):
        super(SimpleConv, self).__init__()

        self.C: int = C
        self.H: int = H
        self.W: int = W
        self.T: int = T

        neuron_params = {
            "threshold": threshold,
            "current_decay": current_decay,
            "voltage_decay": voltage_decay,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
            "requires_grad": requires_grad,
        }
        sdnn_cnn_params = {  # conv layer has additional mean only batch norm
            **neuron_params,  # copy all sdnn_params
            # "norm": slayer.neuron.norm.MeanOnlyBatchNorm,  # mean only quantized batch normalizaton
        }
        sdnn_dense_params = {  # dense layers have additional dropout units enabled
            **sdnn_cnn_params,  # copy all sdnn_cnn_params
            "dropout": slayer.neuron.Dropout(p=0.05),  # neuron dropout
        }

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.Input(neuron_params),
                slayer.block.cuba.Conv(
                    sdnn_cnn_params,
                    C,
                    16,
                    3,
                    padding=1,
                    stride=2,
                    weight_scale=2,
                    weight_norm=True,
                    delay=True,
                ),
                slayer.block.cuba.Conv(
                    sdnn_cnn_params,
                    16,
                    32,
                    3,
                    padding=1,
                    stride=2,
                    weight_scale=2,
                    weight_norm=True,
                    delay=True,
                ),
                slayer.block.cuba.Conv(
                    sdnn_cnn_params,
                    32,
                    64,
                    3,
                    padding=1,
                    stride=2,
                    weight_scale=2,
                    weight_norm=True,
                    delay=True,
                ),
                slayer.block.cuba.Conv(
                    sdnn_cnn_params,
                    64,
                    128,
                    3,
                    padding=1,
                    stride=2,
                    weight_scale=2,
                    weight_norm=True,
                    delay=True,
                ),
                slayer.block.cuba.Flatten(),
                slayer.block.cuba.Dense(sdnn_dense_params, 8192, 128, weight_norm=True, delay=True), # 16384
                slayer.block.cuba.Affine(neuron_params, 128, C, weight_norm=True),
            ]
        )

    def forward(self, spike):
        count = []

        for block in self.blocks:
            spike = block(spike)
            count.append(torch.mean(spike).item())

        return spike, torch.FloatTensor(count).reshape((1, -1)).to(spike.device)

    def grad_flow(self, path):
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, "synapse")]

        return grad

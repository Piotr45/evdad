import lava.lib.dl.bootstrap as bootstrap
import lava.lib.dl.slayer as slayer
import torch


class SimpleDense(torch.nn.Module):
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
        super(SimpleDense, self).__init__()

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
        neuron_params_drop = {
            **neuron_params,
            "dropout": slayer.neuron.Dropout(p=0.1),
        }

        weight_scale = 2
        weight_norm = True

        self.blocks = torch.nn.ModuleList(
            [
                bootstrap.block.cuba.Flatten(),
                # enable affine transform at input
                bootstrap.block.cuba.Dense(
                    neuron_params_drop,
                    self.H * self.W * self.C,
                    512,
                    weight_norm=weight_norm,
                    weight_scale=weight_scale,
                ),
                bootstrap.block.cuba.Dense(
                    neuron_params_drop,
                    512,
                    512,
                    weight_norm=weight_norm,
                    weight_scale=weight_scale,
                ),
                bootstrap.block.cuba.Affine(
                    neuron_params,
                    512,
                    3,
                    weight_norm=weight_norm,
                    weight_scale=weight_scale,
                ),
            ]
        )

    def forward(self, x, mode):
        count = []
        N = x.shape[0]

        for block, m in zip(self.blocks, mode):
            x = block(x, mode=m)
            count.append(torch.mean(x).item())

        x = torch.mean(x, dim=-1).reshape((N, -1))

        return x, torch.FloatTensor(count).reshape((1, -1)).to(x.device)

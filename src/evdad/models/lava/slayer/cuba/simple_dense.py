import lava.lib.dl.slayer as slayer
import torch


class SimpleDense(torch.nn.Module):
    """Simple Dense model class."""

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
            "dropout": slayer.neuron.Dropout(p=0.05),
        }

        weight_norm = True

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.Dense(neuron_params_drop, H * W * C, 256, weight_norm=weight_norm, delay=True),
                slayer.block.cuba.Dense(neuron_params_drop, 256, 128, weight_norm=weight_norm, delay=True),
                slayer.block.cuba.Dense(neuron_params_drop, 128, 64, weight_norm=weight_norm, delay=True),
                slayer.block.cuba.Dense(neuron_params_drop, 64, 32, weight_norm=weight_norm, delay=True),
                slayer.block.cuba.Dense(neuron_params_drop, 32, 16, weight_norm=weight_norm, delay=True),
                slayer.block.cuba.Affine(neuron_params, 16, 10, weight_norm=True),
            ]
        )

    def forward(self, spike: torch.Tensor) -> torch.Tensor:
        count = []

        for block in self.blocks:
            spike = block(spike)
            count.append(torch.mean(spike).item())

        return spike, torch.FloatTensor(count).reshape((1, -1)).to(spike.device)

    def grad_flow(self) -> torch.Tensor:
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, "synapse")]
        return grad

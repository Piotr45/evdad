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

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.Dense(
                    neuron_params_drop, H * W * C, 512, weight_norm=True, delay=True
                ),
                slayer.block.cuba.Dense(
                    neuron_params_drop, 512, 512, weight_norm=True, delay=True
                ),
                slayer.block.cuba.Dense(neuron_params, 512, 3, weight_norm=True),
            ]
        )

    def forward(self, spike: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            spike = block(spike)
        return spike

    def grad_flow(self) -> torch.Tensor:
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, "synapse")]
        return grad

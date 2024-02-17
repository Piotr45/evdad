import lava.lib.dl.slayer as slayer
import torch


class SimpleDense(torch.nn.Module):
    """Simple Dense model class."""

    def __init__(self):
        super(SimpleDense, self).__init__()

        neuron_params = {
            "threshold": 1.25,
            "current_decay": 0.25,
            "voltage_decay": 0.03,
            "tau_grad": 0.03,
            "scale_grad": 3,
            "requires_grad": False,
        }
        neuron_params_drop = {
            **neuron_params,
            "dropout": slayer.neuron.Dropout(p=0.05),
        }
        neuron_params_drop = {**neuron_params}

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.Dense(
                    neuron_params_drop, 260 * 346 * 2, 16, weight_norm=True, delay=True
                ),
                slayer.block.cuba.Dense(
                    neuron_params_drop, 16, 16, weight_norm=True, delay=True
                ),
                slayer.block.cuba.Dense(neuron_params, 16, 3, weight_norm=True),
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

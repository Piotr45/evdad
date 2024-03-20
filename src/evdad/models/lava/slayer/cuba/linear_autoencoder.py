import lava.lib.dl.slayer as slayer
import matplotlib.pyplot as plt
import torch


class LinearAutoEncoder(torch.nn.Module):
    """Linear Auto Encoder model class."""

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
        super(LinearAutoEncoder, self).__init__()

        neuron_params = {
            "threshold": threshold,
            "current_decay": current_decay,
            "voltage_decay": voltage_decay,
            "requires_grad": requires_grad,
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
        }
        neuron_params_drop = {
            **neuron_params,
            "dropout": slayer.neuron.Dropout(p=0.05),
        }

        c_hid = 512
        weight_scale = 1
        weight_norm = False

        self.blocks = torch.nn.ModuleList(
            [
                # Encoder
                slayer.block.cuba.Dense(
                    neuron_params, H * W * C, c_hid, weight_norm=weight_norm, weight_scale=weight_scale, delay=True
                ),
                slayer.block.cuba.Dense(
                    neuron_params, c_hid, c_hid // 2, weight_norm=weight_norm, weight_scale=weight_scale, delay=True
                ),
                slayer.block.cuba.Dense(
                    neuron_params,
                    c_hid // 2,
                    c_hid // 4,
                    weight_norm=weight_norm,
                    weight_scale=weight_scale,
                    delay=True,
                ),
                slayer.block.cuba.Dense(
                    neuron_params,
                    c_hid // 4,
                    c_hid // 8,
                    weight_norm=weight_norm,
                    weight_scale=weight_scale,
                    delay=True,
                ),
                slayer.block.cuba.Dense(
                    neuron_params,
                    c_hid // 8,
                    c_hid // 16,
                    weight_norm=weight_norm,
                    weight_scale=weight_scale,
                    delay=True,
                ),
                slayer.block.cuba.Dense(
                    neuron_params,
                    c_hid // 16,
                    c_hid // 32,
                    weight_norm=weight_norm,
                    weight_scale=weight_scale,
                    delay=True,
                ),
                # Decoder
                slayer.block.cuba.Dense(
                    neuron_params,
                    c_hid // 32,
                    c_hid // 16,
                    weight_norm=weight_norm,
                    weight_scale=weight_scale,
                    delay=True,
                ),
                slayer.block.cuba.Dense(
                    neuron_params,
                    c_hid // 16,
                    c_hid // 8,
                    weight_norm=weight_norm,
                    weight_scale=weight_scale,
                    delay=True,
                ),
                slayer.block.cuba.Dense(
                    neuron_params,
                    c_hid // 8,
                    c_hid // 4,
                    weight_norm=weight_norm,
                    weight_scale=weight_scale,
                    delay=True,
                ),
                slayer.block.cuba.Dense(
                    neuron_params,
                    c_hid // 4,
                    c_hid // 2,
                    weight_norm=weight_norm,
                    weight_scale=weight_scale,
                    delay=True,
                ),
                slayer.block.cuba.Dense(
                    neuron_params, c_hid // 2, c_hid, weight_norm=weight_norm, weight_scale=weight_scale, delay=True
                ),
                slayer.block.cuba.Dense(
                    neuron_params, c_hid, H * W * C, weight_norm=weight_norm, weight_scale=weight_scale, delay=True
                ),
            ]
        )

    def forward(self, spike: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            spike = block(spike)
        return spike

    def grad_flow(self) -> torch.Tensor:
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, "synapse")]
        plt.figure()
        plt.semilogy(grad)
        plt.savefig("./gradFlow.png")
        plt.close()
        return grad

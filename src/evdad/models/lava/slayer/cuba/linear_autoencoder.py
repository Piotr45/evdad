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

        self.C: int = C
        self.H: int = H
        self.W: int = W
        self.T: int = T

        # neuron_params = {
        #     "threshold": threshold,
        #     "current_decay": current_decay,
        #     "voltage_decay": voltage_decay,
        #     "requires_grad": requires_grad,
        #     "tau_grad": tau_grad,
        #     "scale_grad": scale_grad,
        #     "graded_spike": False,
        # }
        neuron_params = {
                'threshold'     : 0.1,
                'current_decay' : 1,
                'voltage_decay' : 0.1,
                'requires_grad' : True,     
            }
        neuron_params_drop = {
            **neuron_params,
            "dropout": slayer.neuron.Dropout(p=0.05),
            "norm": slayer.neuron.norm.MeanOnlyBatchNorm,
        }

        c_hid = 2048
        weight_scale = 2
        weight_norm = True
        delay = True

        self.blocks = torch.nn.ModuleList(
            [
                # Encoder
                slayer.block.cuba.Dense(
                    neuron_params,
                    H * W * C,
                    c_hid,
                    weight_norm=weight_norm,
                    weight_scale=weight_scale,
                    delay=delay,
                ),
                # Decoder
                slayer.block.cuba.Dense(
                    neuron_params,
                    c_hid,
                    H * W * C,
                    weight_norm=weight_norm,
                    weight_scale=weight_scale,
                ),
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
        plt.figure()
        plt.semilogy(grad)
        plt.savefig("./gradFlow.png")
        plt.close()
        return grad

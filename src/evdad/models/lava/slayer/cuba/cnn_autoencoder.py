import lava.lib.dl.slayer as slayer
import matplotlib.pyplot as plt
import torch


class CNNAutoEncoder(torch.nn.Module):
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
        super(CNNAutoEncoder, self).__init__()

        self.encoder = Encoder(
            C,
            2,
            threshold,
            current_decay,
            voltage_decay,
            tau_grad,
            scale_grad,
            requires_grad,
        )
        self.decoder = Decoder(
            C,
            2,
            threshold,
            current_decay,
            voltage_decay,
            tau_grad,
            scale_grad,
            requires_grad,
        )

    def forward(self, spike: torch.Tensor) -> torch.Tensor:
        # print("encoder")
        z = self.encoder(spike)
        # print("between")
        x_hat, c = self.decoder(z)
        # print("xhat", x_hat.shape)
        return x_hat, c

    def grad_flow(self) -> torch.Tensor:
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, "synapse")]
        plt.figure()
        plt.semilogy(grad)
        plt.savefig("./gradFlow.png")
        plt.close()
        return grad


class Encoder(torch.nn.Module):
    def __init__(
        self,
        C: int,
        base_channel_size: int,
        threshold: float,
        current_decay: float,
        voltage_decay: float,
        tau_grad: float,
        scale_grad: float,
        requires_grad: bool,
    ) -> None:
        super().__init__()

        neuron_params = {
            "threshold": threshold,
            "current_decay": current_decay,
            "voltage_decay": voltage_decay,
            "requires_grad": requires_grad,
        }
        neuron_params_drop = {
            **neuron_params,
            # "dropout": slayer.neuron.Dropout(p=0.05),
        }

        c_hid = base_channel_size

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.Conv(
                    neuron_params_drop,
                    C,
                    c_hid,
                    3,
                    padding=1,
                    stride=2,
                    weight_scale=2,
                    weight_norm=True,
                    delay=True,
                ),
            ]
        )

    def forward(self, spike: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            spike = block(spike)
        return spike


class Decoder(torch.nn.Module):
    def __init__(
        self,
        C: int,
        base_channel_size: int,
        threshold: float,
        current_decay: float,
        voltage_decay: float,
        tau_grad: float,
        scale_grad: float,
        requires_grad: bool,
    ) -> None:
        super().__init__()

        neuron_params = {
            "threshold": threshold,
            "current_decay": current_decay,
            "voltage_decay": voltage_decay,
            "requires_grad": requires_grad,
        }
        neuron_params_drop = {
            **neuron_params,
            # "dropout": slayer.neuron.Dropout(p=0.05),
        }

        c_hid = base_channel_size
        weight_scale = 1

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.ConvT(
                    neuron_params_drop,
                    c_hid,
                    C,
                    3,
                    padding=0,
                    stride=2,
                    weight_scale=weight_scale,
                    weight_norm=True,
                    delay=False,
                    # dilation=0,
                ),
                slayer.block.cuba.Conv(
                    neuron_params_drop,
                    C,
                    C,
                    2,
                    padding=0,
                    stride=1,
                    weight_scale=weight_scale,
                    weight_norm=False,
                    delay=False,
                ),
            ]
        )

    def forward(self, spike: torch.Tensor) -> torch.Tensor:
        count = []

        for block in self.blocks:
            spike = block(spike)
            count.append(torch.mean(spike).item())

        return spike, torch.FloatTensor(count).reshape((1, -1)).to(spike.device)

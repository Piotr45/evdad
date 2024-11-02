import h5py
import lava.lib.dl.slayer as slayer
import matplotlib.pyplot as plt
import torch

from evdad.models.lava.slayer.interface import EVDADModel


class CNNAutoEncoder(EVDADModel):
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

        self.C: int = C
        self.H: int = H
        self.W: int = W
        self.T: int = T

        self.encoder = Encoder(
            C,
            32,
            threshold,
            current_decay,
            voltage_decay,
            tau_grad,
            scale_grad,
            requires_grad,
        )
        self.decoder = Decoder(
            C,
            32,
            threshold,
            current_decay,
            voltage_decay,
            tau_grad,
            scale_grad,
            requires_grad,
        )

    def forward(self, spike: torch.Tensor) -> torch.Tensor:
        org = torch.tensor(torch.mean(spike).item()).reshape((1, -1)).to("cuda")
        z, ec = self.encoder(spike)
        x_hat, c = self.decoder(z)

        # TODO do it in model
        N, C, H, W, T = x_hat.shape
        zeros = torch.zeros((N, C, spike.shape[2], spike.shape[3], T))
        zeros[:, :, :H, :W, :] = x_hat[:, :, :H, :W, :]

        return zeros.to("cuda"), torch.concat((ec, c, org), dim=1)
        # return x_hat, torch.concat((ec, c, org), dim=1)

    def grad_flow(self) -> torch.Tensor:
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, "synapse")]
        plt.figure()
        plt.semilogy(grad)
        plt.savefig("./gradFlow.png")
        plt.close()
        return grad

    def export_hdf5(self, filename: str) -> None:
        # network export to hdf5 format
        h = h5py.File(filename, "w")
        layer = h.create_group("layer")
        c = 0
        for i, b in enumerate(self.encoder.blocks):
            b.export_hdf5(layer.create_group(f"{i}"))
            c = i
        c += 1
        for i, b in enumerate(self.decoder.blocks):
            b.export_hdf5(layer.create_group(f"{c+i}"))


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
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
        }
        neuron_params_drop = {
            **neuron_params,
            # "dropout": slayer.neuron.Dropout(p=0.05),
        }

        c_hid = base_channel_size
        weight_scale = 2
        weight_norm = False

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.Conv(
                    neuron_params_drop,
                    in_features=C,
                    out_features=c_hid,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    weight_scale=weight_scale,
                    weight_norm=weight_norm,
                    delay=False,
                ),
                slayer.block.cuba.Conv(
                    neuron_params_drop,
                    in_features=c_hid,
                    out_features=c_hid * 2,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    weight_scale=weight_scale,
                    weight_norm=weight_norm,
                    delay=False,
                ),
                slayer.block.cuba.Conv(
                    neuron_params_drop,
                    in_features=c_hid * 2,
                    out_features=c_hid * 4,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    weight_scale=weight_scale,
                    weight_norm=weight_norm,
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
            "tau_grad": tau_grad,
            "scale_grad": scale_grad,
        }
        neuron_params_drop = {
            **neuron_params,
            # "dropout": slayer.neuron.Dropout(p=0.05),
        }

        c_hid = base_channel_size
        weight_scale = 2
        weight_norm = False

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.ConvT(
                    neuron_params_drop,
                    in_features=c_hid * 4,
                    out_features=c_hid * 2,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    weight_scale=weight_scale,
                    weight_norm=weight_norm,
                    delay=False,
                ),
                slayer.block.cuba.ConvT(
                    neuron_params_drop,
                    in_features=c_hid * 2,
                    out_features=c_hid,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    weight_scale=weight_scale,
                    weight_norm=weight_norm,
                    delay=False,
                ),
                slayer.block.cuba.ConvT(
                    neuron_params_drop,
                    in_features=c_hid,
                    out_features=C,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    weight_scale=weight_scale,
                    weight_norm=weight_norm,
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

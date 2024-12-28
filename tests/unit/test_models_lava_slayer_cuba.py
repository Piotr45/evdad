import numpy as np
import torch

from evdad.models.lava.slayer.cuba.cnn_autoencoder import CNNAutoEncoder
from evdad.models.lava.slayer.cuba.linear_autoencoder import LinearAutoEncoder
from evdad.models.lava.slayer.cuba.simple_conv import SimpleConv
from evdad.models.lava.slayer.cuba.simple_dense import SimpleDense
from evdad.utils import set_seed


def test_model_simple_conv():
    C, H, W, T = 2, 128, 128, 30
    threshold = 0.1
    current_decay = 0.3
    voltage_decay = 0.03
    tau_grad = 0.01
    scale_grad = 3
    requires_grad = False

    model = SimpleConv(C, H, W, T, threshold, current_decay, voltage_decay, tau_grad, scale_grad, requires_grad)

    assert model.C == C
    assert model.H == H
    assert model.W == W
    assert model.T == T


def test_model_simple_conv_forward():
    set_seed(3108)
    device = torch.device("cpu")
    C, H, W, T = 2, 128, 128, 30
    num_classes = 3
    threshold = 0.1
    current_decay = 0.3
    voltage_decay = 0.03
    tau_grad = 0.01
    scale_grad = 3
    requires_grad = False

    model = SimpleConv(C, H, W, T, threshold, current_decay, voltage_decay, tau_grad, scale_grad, requires_grad)

    test_tensor = torch.randn(1, C, H, W, T, device=device, dtype=torch.float32)
    output_tensor, _ = model.forward(test_tensor)
    output_tensor = output_tensor.detach().cpu().numpy()

    assert output_tensor.shape == (1, num_classes, T)
    assert np.allclose(output_tensor[0, 0, 5:10], np.array([0.0, 0.0, 0.0, 0.0, 0.0]))


def test_model_simple_dense():
    C, H, W, T = 2, 128, 128, 30
    threshold = 0.1
    current_decay = 0.3
    voltage_decay = 0.03
    tau_grad = 0.01
    scale_grad = 3
    requires_grad = False

    model = SimpleDense(C, H, W, T, threshold, current_decay, voltage_decay, tau_grad, scale_grad, requires_grad)

    assert model.C == C
    assert model.H == H
    assert model.W == W
    assert model.T == T


def test_model_simple_dense_forward():
    set_seed(3108)
    device = torch.device("cpu")
    num_classes = 3
    C, H, W, T = 2, 128, 128, 30
    threshold = 0.1
    current_decay = 0.3
    voltage_decay = 0.03
    tau_grad = 0.01
    scale_grad = 3
    requires_grad = False

    model = SimpleDense(C, H, W, T, threshold, current_decay, voltage_decay, tau_grad, scale_grad, requires_grad)

    test_tensor = torch.randn(1, C, H, W, T, device=device, dtype=torch.float32)
    output_tensor, _ = model.forward(test_tensor)
    output_tensor = output_tensor.detach().cpu().numpy()

    assert output_tensor.shape == (1, num_classes, T)
    assert np.allclose(output_tensor[0, 0, 5:10], np.array([0.0, 0.0, 0.0, 0.0, 0.0]))


def test_model_cnn_ae():
    C, H, W, T = 2, 128, 128, 30
    threshold = 0.1
    current_decay = 0.3
    voltage_decay = 0.03
    tau_grad = 0.01
    scale_grad = 3
    requires_grad = False

    model = CNNAutoEncoder(C, H, W, T, threshold, current_decay, voltage_decay, tau_grad, scale_grad, requires_grad)

    assert model.C == C
    assert model.H == H
    assert model.W == W
    assert model.T == T


def test_model_cnn_ae_forward():
    set_seed(3108)
    device = torch.device("cpu")
    C, H, W, T = 2, 128, 128, 30
    threshold = 0.1
    current_decay = 0.3
    voltage_decay = 0.03
    tau_grad = 0.01
    scale_grad = 3
    requires_grad = False

    model = CNNAutoEncoder(C, H, W, T, threshold, current_decay, voltage_decay, tau_grad, scale_grad, requires_grad)

    test_tensor = torch.randn(1, C, H, W, T, device=device, dtype=torch.float32)
    output_tensor, _ = model.forward(test_tensor)
    output_tensor = output_tensor.detach().cpu().numpy()

    assert output_tensor.shape == (1, C, H, W, T)
    assert np.allclose(output_tensor[0, 0, 51, 51, 5:10], np.array([0.0, 0.0, 0.0, 0.0, 0.0]))


def test_model_linear_ae():
    C, H, W, T = 2, 128, 128, 30
    threshold = 0.1
    current_decay = 0.3
    voltage_decay = 0.03
    tau_grad = 0.01
    scale_grad = 3
    requires_grad = False

    model = LinearAutoEncoder(C, H, W, T, threshold, current_decay, voltage_decay, tau_grad, scale_grad, requires_grad)

    assert model.C == C
    assert model.H == H
    assert model.W == W
    assert model.T == T


def test_model_linear_ae_forward():
    set_seed(3108)
    device = torch.device("cpu")
    C, H, W, T = 2, 128, 128, 1
    threshold = 0.1
    current_decay = 0.3
    voltage_decay = 0.03
    tau_grad = 0.01
    scale_grad = 3
    requires_grad = False

    model = LinearAutoEncoder(C, H, W, T, threshold, current_decay, voltage_decay, tau_grad, scale_grad, requires_grad)

    test_tensor = torch.randn(C, H, W, device=device, dtype=torch.float32).flatten().unsqueeze(0).unsqueeze(-1)
    output_tensor, _ = model.forward(test_tensor)
    output_tensor = output_tensor.detach().cpu().numpy()

    assert output_tensor.shape == (1, C * H * W, T)
    assert np.allclose(output_tensor[0, 5000:5005, 0], np.array([0.0, 0.0, 0.0, 0.0, 0.0]))


test_model_linear_ae_forward()

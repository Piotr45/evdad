# Models

Neuron models used in the project:

1. [Slayer](https://lava-nc.org/lava-lib-dl/slayer/slayer.html)
    - [ALIF](https://lava-nc.org/lava-lib-dl/slayer/slayer.html)
    - [CUBA](https://lava-nc.org/lava-lib-dl/slayer/slayer.html)
    - [Sigma Delta](https://lava-nc.org/lava-lib-dl/slayer/slayer.html)
2. [Bootstrap](https://lava-nc.org/lava-lib-dl/bootstrap/bootstrap.html)

## Slayer

### Adaptive Leaky Integrate and Fire

Currently not supported via predefined model.

### CUrrent BAsed leaky integrator

This is the implementation of Loihi CUBA neuron.

$$
u[t] = (1 - \alpha_u) \, u[t-1] + x[t],
$$
$$
v[t] = (1 - \alpha_v) \, v[t-1] + u[t] + \text{bias},
$$
$$
s[t] = v[t] \geq \vartheta,
$$
$$
v[t] = v[t] \, (1 - s[t]).
$$

The internal state representations are scaled down compared to the actual hardware implementation. This allows for a natural range of synaptic weight values as well as the gradient parameters.

Source: [lava docs](https://lava-nc.org/lava-lib-dl/slayer/neuron/neuron.html#module-lava.lib.dl.slayer.neuron.cuba).

### Sigma Delta 

Sigma-delta neural networks consists of two main units: sigma decoder in the dendrite and delta encoder in the axon. Delta encoder uses differential encoding on the output activation of a regular ANN activation, for e.g. ReLU. In addition it only sends activation to the next layer when the encoded message magnitude is larger than its threshold. The sigma unit accumulates the sparse event messages and accumulates it to restore the original value.

A sigma-delta neuron is simply a regular activation wrapped around by a sigma unit at it's input and a delta unit at its output.

When the input to the network is a temporal sequence, the activations do not change much. Therefore, the message between the layers are reduced which in turn reduces the synaptic computation in the next layer. In addition, the graded event values can encode the change in magnitude in one time-step. Therefore there is no increase in latency at the cost of time-steps unlike the rate coded Spiking Neural Networks.

Source: [lava-dl](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/pilotnet/train.ipynb).

## Bootstrap

In general ANN-SNN conversion methods for rate based SNN result in high latency of the network during inference. This is because the rate interpretation of a spiking neuron using ReLU acitvation unit breaks down for short inference times. As a result, the network requires many time steps per sample to achieve adequate inference results.

Bootstrap (lava.lib.dl.bootstrap) enables rapid training of rate based SNNs by translating them to an equivalent dynamic ANN representation which leads to SNN performance close to the equivalent ANN and low latency inference. More details here. It also supports hybrid training a mixed ANN-SNN network to minimize the ANN to SNN performance gap. This method is independent of the SNN model being used.

It has similar API as lava.lib.dl.slayer and supports exporting trained models using the platform independent hdf5 network exchange format.

Source: [lava docs](https://lava-nc.org/dl.html#bootstrap).

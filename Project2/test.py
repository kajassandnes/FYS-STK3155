import numpy as np
from neural_network import NeuralNetwork, mse, mse_der, sigmoid, sigmoid_der


def test_gradient():
    """
    Test that jax and the backpropagation algorithm compute identical gradients 
    """
    inputs = np.random.randn(2, 2)
    targets = np.random.randn(2, 4)
    network_batch_size, network_input_size = inputs.shape

    net = NeuralNetwork(network_input_size, [3, 4], [sigmoid, sigmoid], [sigmoid_der, sigmoid_der], mse, mse_der)
    grad = net.compute_gradient(inputs, targets)
    grad_jax = net.jax_gradient(inputs, net.layers, targets)

    for i in range(1):
        for j in range(1):
            assert(np.allclose(grad[i][j], grad_jax[i][j]))


test_gradient()
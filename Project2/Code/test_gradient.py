import numpy as np
from neural_network import NeuralNetwork, mse, mse_der, sigmoid, sigmoid_der
from utils import mse, mse_der, sigmoid, sigmoid_der


def test_gradient():
    """
    Test that jax and the backpropagation algorithm compute identical gradients 
    """
    inputs = np.random.randn(2, 2)
    targets = np.random.randn(2, 4)
    network_batch_size, network_input_size = inputs.shape

    net = NeuralNetwork(network_input_size, [3, 4], [sigmoid, sigmoid], [sigmoid_der, sigmoid_der], mse, mse_der, lmbda=0.0)
    grad = net.backpropagation(inputs, targets)
    grad_jax = net.jax_gradient(inputs, net.layers, targets)

    for (W, b), (W_jax, b_jax) in zip(grad, grad_jax):
        assert np.allclose(W, W_jax)
        assert np.allclose(b, b_jax)
    
    print("Test passed!")
    

if __name__ == "__main__":
    test_gradient()

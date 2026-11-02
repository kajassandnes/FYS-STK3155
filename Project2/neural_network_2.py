# class for neural network
import numpy as np

class Network():
    """Network is a class used for traning neural networks.

    Public methods:
    SGD() - Stochastic gradient descent

    Private methods:
    __init__() 
    _normal_initialize_weights_and_biases()
    _backpropagation()
    _feedforward()

    The networks is inspired by methods in the books:
    - Neural Networks and Deep learning by Michael Nielsen (2019)
      http://neuralnetworksanddeeplearning.com/index.html
    - Machine Learning and the Physical Sciences by Morten Hjort-Jensen (2026)
      https://github.com/EducationalMaterialUiO/MachineLearningUiO/blob/main/doc/BookML/Lecturebook.pdf

    Use of LLM:

    """
    def __init__(self, layers:list, activation_func:callable, activation_der: callable, 
                 cost_func:callable, cost_der:callable, output_func:callable, 
                 output_der:callable, init="", seed=2026):
        """
        All callables are the users choice of costfunction and activation function, and 
        their respective derivatives. Having the output function as an argument 
        generalizes the network to be able to do regression and classification. 
        init lets you decide how to initialize the weights. Default is Xavier (Glorot).
        layers is a list with structure [input layer, hidden layers, output layer].
        """
        self.layers = layers
        self.activation_func = activation_func
        self.activation_der = activation_der
        self.cost_func = cost_func
        self.cost_der = cost_der
        self.output_func = output_func
        self.output_der = output_der
        self.rng = np.random.default_rng(seed)
        self.init = init
        self._normal_initialize_weights_and_biases()

    def _normal_initialize_weights_and_biases(self) -> None:
        """Initialises all weights and biases from a normal distribution with mean=0 and
        He initialisation: std=sqrt(2/M_{l-1}) if init = "He".
        Xavier (Glorot) initialization: std=sqrt(2/(M_{l-1} + M_l)) else.
        Weights W[l] in layer l have shape (n_in, n_out), biases B[l] have shape (n_out,)
        """
        pairs = list(zip(self.layers[:-1], self.layers[1:]))
        if self.init.lower() == "he":
            self.W = [self.rng.normal(loc=0, scale=np.sqrt(2/(n_in)), size=(n_in, n_out)) 
                      for n_in, n_out in pairs]
            self.B = [np.ones(n_out)*0.01 for _, n_out in pairs]

        else:
            self.W = [self.rng.normal(loc=0, scale=np.sqrt(2/(n_in + n_out)), size=(n_in, n_out)) 
                      for n_in, n_out in pairs]
            self.B = [np.zeros(n_out) for _, n_out in pairs]
    
    def _feedforward(self, X:list) -> list:
        """Feedforward is used to calculate activations and weighted inputs by
        1. looping through all layers
        2. calculating weighted inputs by z^l = w^l * a^{l-1} + b^l
        3. calculating activations by a^l = S(z^l), where S is the activation
           function
        4. storing all activations and weighted inputs in lists

        X (n_samples, n_features) is the input data. 
        """
        a = X
        zs, activations = [], [X]
        # looping through each layer and calculating a and z
        # for all nodes at the same time using linearization
        for l in range(len(self.W)):
            z = a @ self.W[l] + self.B[l]
            a = self.output_func(z) if l == len(self.W) - 1 else self.activation_func(z)
            zs.append(z)
            activations.append(a)
        return zs, activations

    def _backpropagation(self, X:list, y:list) -> list:
        """Backpropagation is used to calculate gradients by
        1. using feedforward to calculate all activations and weighted inputs
        2. using backward pass to calculate all errors from the last to the 
           first layer
        3. calculating all derivatives of the cost with respect to all weights
           and biases

        y (n_samples, n_outputs)
        X (n_samples, n_inputs) 
        
        where n_samples is the length of each minibatch.
        """
        zs, activations = self._feedforward(X)
        n = len(X)
        grad_W = [None] * len(self.W) # arrays cannot store matrices, therefore [None]
        grad_B = [None] * len(self.B) 

        delta = self.cost_der(activations[-1], y) * self.output_der(zs[-1])      # error in the last layer

        for l in range(1, len(self.W) + 1):                  # l = 1 is the last layer
            grad_W[-l] = activations[-l - 1].T @ delta / n  
            grad_B[-l] = delta.mean(axis=0)
            if l < len(self.W):                            
                delta = (delta @ self.W[-l].T) * self.activation_der(zs[-l - 1]) # error one layer back

        return grad_W, grad_B

    def SGD(self, X, y, n_epochs, m, eta):
        """Stochastic gradient descent trains the network by
        1. looping through n_epochs
        2. splitting the input data into random minibatches of size m
        3. using backpropagation to calculate gradients for each weight and bias
        4. updating all weights and biases

        X (n_samples, n_input)   - input data
        y (n_samples, n_output)  - target
        n_epochs (int)           - number of epochs
        m (int)                  - size of minibatch
        eta (float)              - learning rate
        """
        n = len(X)
        for epoch in range(n_epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, m):
                batch = idx[start:start + m]
                grad_W, grad_B = self._backpropagation(X[batch], y[batch])
                for l in range(len(self.W)):
                    self.W[l] -= eta * grad_W[l]
                    self.B[l] -= eta * grad_B[l]
        pass


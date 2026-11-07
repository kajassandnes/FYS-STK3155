import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that results agree with numpy to ~1e-15
import optax
from utils import *
from sklearn.base import BaseEstimator


# NeuralNetwork class:

class NeuralNetwork:
    def __init__( self, network_input_size, layer_output_sizes, activation_funcs,
                  activation_ders, cost_fun, cost_der, lmbda, seed = 2026 ):
        """Description"""       
        self.network_input_size = network_input_size    
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.lmbda = lmbda
        self.seed = seed
        self._initialize_layers()

    def __str__(self) -> "str":
        return(f"Neural network with {len(self.layer_output_sizes) - 1} hidden layers.")

    def _initialize_layers(self, init=None) -> None:
        """Initialises all weights and biases from a normal distribution with mean zero and
        either He or Xavier initialisation. 

        He initialisation:              std=sqrt( 2 /  M_{l-1} ) if init = "He".
        Xavier (Glorot) initialization: std=sqrt( 2 / (M_{l-1} + M_l) ) else.

        Layers are stored as a list of (W, b), W: (n_in, n_out), b: (n_out,)
        """
        rng = np.random.default_rng(self.seed)
        sizes = [self.network_input_size] + list(self.layer_output_sizes)
        pairs = list(zip(sizes[:-1], sizes[1:]))
        
        if init == "He":
            W = [rng.normal(loc=0, scale=np.sqrt(2/n_in), size=(n_in, n_out)) for n_in, n_out in pairs]
            B = [np.ones(n_out) * 0.01 for _, n_out in pairs]
        else:
            W = [rng.normal(loc=0, scale=np.sqrt(2/(n_in + n_out)), size=(n_in, n_out)) for n_in, n_out in pairs]
            B = [np.zeros(n_out) for _, n_out in pairs]

        self.layers = list(zip(W, B))

    def _predict(self, layers, inputs):
        """
        Complete feedforward pass and predict
        """
        a = inputs
        for (W, b), activation_func in zip(layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)

        return a
    
    def _cost(self, layers, inputs, targets):
        """
        Compute cost via the object's cost function on desired inputs and targets
        """
        a = self._predict(layers, inputs)

        return self.cost_fun(a, targets)

    def predict(self, inputs):                  
        return self._predict(self.layers, inputs)

    def cost(self, inputs, targets):         
        return self._cost(self.layers, inputs, targets)

    def jax_gradient(self, inputs, layers,  targets):
        """
        Compute gradient with automatic differentiation via jax.
        layers contain weights and biases, which are the parameters
        differentiated with respect to.
        """
        grad = jax.grad(self._cost)
        return grad(layers, inputs, targets)

    def _feedforward(self, inputs):
        """
        Complete feed forward pass that saves intermediary values
        Private method used internally for backpropagation
        """
        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)

            zs.append(z)

        return layer_inputs, zs, a

    def backpropagation(self, inputs, targets):
        """
        Computes gradient via analytical backpropagation
        """
        layer_inputs, zs, predict = self._feedforward(inputs)

        layer_grads = [() for _ in self.layers]

        # We loop over the layers, from the last to the first
        for i in reversed(range(len(self.layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_ders[i]
            W_current = self.layers[i][0]

            if i == len(self.layers) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = self.cost_der(predict, targets)
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W, b) = self.layers[i + 1]
                dC_da = dC_dz @ W.T

            dC_dz = dC_da * activation_der(z)
            dC_dW = layer_input.T @ dC_dz + self.lmbda * W_current
            dC_db = np.sum(dC_dz, axis = 0)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads


    def _make_batches(self, n, batch_size, rng):
        """
        Shuffle the indices and split them into minibatches. Private method used in sgd
        """
        idx = rng.permutation(n)
        return [idx[i:i + batch_size] for i in range(0, n, batch_size)]

    def train_sgd(self, inputs, targets, optimizer:optax.GradientTransformation, n_epochs = 50, batch_size = 5, 
                  autodiff = False, report_progress = False) -> None:
        """
        Trains network using stochastic gradient descent.
        optimizer (optax.GradientTransformation): already initialized optimizer
        Gradually updates self.layers, returns None
        """
        layers_optax = self.layers
        opt_state = optimizer.init(layers_optax)
        n_datapoints = inputs.shape[0]
        rng = np.random.default_rng(self.seed)

        for _ in range(n_epochs):
            if report_progress:
                print(f"Epoch: {_ + 1}")
            for batch in self._make_batches(n_datapoints, batch_size, rng):
                layers_grad = self.backpropagation(inputs[batch], targets[batch])
                updates, opt_state = optimizer.update(layers_grad, opt_state, layers_optax)
                layers_optax = optax.apply_updates(layers_optax, updates)

                self.layers = layers_optax

    def train_pgd(self, inputs, targets, optimizer, n_epochs = 100, autodiff = False, report_progress = False):
        """
        Trains network with plain gradient descent.
        optimizer (optax.GradientTransformation): already initialized optimizer
        Gradually updatater self.layers, returns None
        """
        layers_optax = self.layers
        opt_state = optimizer.init(layers_optax)
        
        for _ in range(n_epochs):
            if report_progress:
                print(f"Epoch: {_ + 1}")

            layers_grad = self.backpropagation(inputs, targets)
            updates, opt_state = optimizer.update(layers_grad, opt_state, layers_optax)
            layers_optax = optax.apply_updates(layers_optax, updates)

            self.layers = layers_optax
          
    def save_layers(self):
        """
        Method used to save layers for later validation so layers can be altered.
        """
        return self.layers


class EstimatorClassifier(NeuralNetwork, BaseEstimator):
    """
    EstimatorClassifier class, made specifically to make the code of part e)
    easier to write, and more readable. Has inheretance from BaseEstimator
    and our own NeuralNetwork. Has fixed size of one layer as that is what
    we worked with for part e.
    """
    def __init__( self, hidden = 64, activation_funcs = (ReLU, softmax), activation_ders = (ReLU_der, lambda _: 1),
                  cost_fun = cross_entropy, cost_der = smce_der, seed = 2026, gamma = 0.03, epochs = 10,
                  batch_size = 10, lmbda = 0.0 ):
        """
        Description
        """
        self.hidden = hidden
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.seed = seed
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.lmbda = lmbda

    def fit(self, X, y):
        """
        Sklearn compliant classifier fit method.
        """
        self.classes_ = np.unique(y)
        Y = one_hot(np.searchsorted(self.classes_, y), len(self.classes_))
        self.network_input_size = X.shape[1]
        self.layer_output_sizes = (self.hidden, len(self.classes_))
        self._initialize_layers(init="He")
        optimizer = optax.adam(learning_rate = self.gamma)
        self.train_sgd(X, Y, optimizer, self.epochs, self.batch_size)
        self.layers_ = self.layers
        return self

    def cost(self, inputs, targets):
        """
        Compute cost via the object's cost function on desired inputs and targets
        """
        a = self.predict(inputs)
        Y = one_hot(np.searchsorted(self.classes_, targets), len(self.classes_))
        return self.cost_fun(a, Y)

    def predict_proba(self, X):
        """
        Returns matrix of probabilities for each class for each input.
        """
        return super().predict(X)
    
    def predict(self, X):
        """
        Returns vector of predictions chosen as largest softmax probability.
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis = 1)]

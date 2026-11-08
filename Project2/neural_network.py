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

    def _cost_with_penalty(self, layers, inputs, targets):
        """Function is used for testing gradients against 
        automatic differentiation with jax.
        """
        c = self._cost(layers, inputs, targets)
        ws = [W for W, b in layers]

        if self.penalty == "l1":
            return c + self.lmbda * sum(jnp.sum(jnp.abs(W)) for W in ws)
        else:
            return c + 0.5 * self.lmbda * sum(jnp.sum(W**2) for W in ws)

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
        grad = jax.grad(self._cost_with_penalty)
        return grad(layers, inputs, targets)
    
    def _penalty_grad(self, W):
        """Adds regularization term."""
        if self.lmbda == 0:
            return 0.0
        elif self.penalty == "l1":
            return self.lmbda * np.sign(W)
        else:
            return self.lmbda * W          

    def _feedforward(self, inputs):
        """
        Feedforward is used to calculate activations and weighted inputs by
        1. looping through all layers
        2. calculating weighted inputs by z^l = a^{l-1} * w^l + b^l
        3. calculating activations by a^l = sigma(z^l), where sigma is the activation
           function
        4. storing all activations and weighted inputs in lists
        5. returns activations and weighted activations

        inputs (n_samples, n_features): input data. 
        """
        a = inputs
        zs, activations = [], [inputs]

        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
            activations.append(a)

        return zs, activations

    def backpropagation(self, inputs, targets):
        """
        Backpropagation is used to calculate gradients by
        1. using feedforward to calculate all activations and weighted inputs
        2. using backward pass to calculate all errors from the last to the 
           first layer
        3. calculating all derivatives of the cost with respect to all weights
           and biases

        targets (n_samples, n_outputs) : target
        inputs (n_samples, n_inputs)  : input data
        
        where n_samples is the length of each minibatch when using plain gradient descent.
        """
        zs, activations = self._feedforward(inputs)
        grads = [None] * len(self.layers)

        # error in the last layer
        delta = self.cost_der(activations[-1], targets) * self.activation_ders[-1](zs[-1])

        for l in reversed(range(len(self.layers))):
            W, b = self.layers[l]
            grads[l] = (activations[l].T @ delta + self._penalty_grad(W), delta.sum(axis=0))

            if l > 0:                       # error one layer back
                delta = (delta @ W.T) * self.activation_ders[l - 1](zs[l - 1])

        return grads


    def _make_batches(self, n, batch_size, rng):
        """
        Shuffle the indices and split them into minibatches. Private method used in sgd
        """
        idx = rng.permutation(n)
        return [idx[i:i + batch_size] for i in range(0, n, batch_size)]

    def train_sgd(self, inputs, targets, optimizer:optax.GradientTransformation, 
                  n_epochs = 50, batch_size = 5, report_progress = False, 
                  X_val=None, y_val=None, record_every=None) -> dict:
        """Stochastic gradient descent trains the network by
        1. looping through n_epochs
        2. splitting the input data into random minibatches of size m
        3. using backpropagation to calculate gradients for each weight and bias
        4. updating all weights and biases
        5. optionally saves all cost from training and validation data

        inputs (n_samples, n_input)     - input data
        targets (n_samples, n_output)   - target
        optimizer (optax.GradientTransformation) - already initialized optimizer
        n_epochs (int)                  - number of epochs
        batch_size (int)                - size of minibatch
        report_progress (bool)          - prints epoch
        X_val (n_val_samples, n_input)  - validation data
        X_val (n_val_samples, n_output) - validation data
        record_every (int)              - recording frequency        
        """
        layers_optax = self.layers
        opt_state = optimizer.init(layers_optax)
        n_datapoints = inputs.shape[0]
        rng = np.random.default_rng(self.seed)
        history = {"epoch": [], "train": [], "val": []}

        for epoch in range(1, n_epochs + 1):
            if report_progress:
                print(f"Epoch: {epoch}")
            for batch in self._make_batches(n_datapoints, batch_size, rng):
                layers_grad = self.backpropagation(inputs[batch], targets[batch])
                updates, opt_state = optimizer.update(layers_grad, opt_state, layers_optax)
                layers_optax = optax.apply_updates(layers_optax, updates)

                self.layers = layers_optax

            if record_every and epoch % record_every == 0:
                history["epoch"].append(epoch)
                history["train"].append(float(self.cost(inputs, targets)))
                history["val"].append(float(self.cost(X_val, y_val)) if X_val is not None else None)
        
        return history

    def train_pgd(self, inputs, targets, optimizer:optax.GradientTransformation,
                  n_epochs = 100, report_progress = False, X_val=None, y_val=None, 
                  record_every=None) -> dict:
        """
        Trains network with plain gradient descent.
        optimizer (optax.GradientTransformation): already initialized optimizer
        Gradually updatater self.layers, returns None
        """
        layers_optax = self.layers
        opt_state = optimizer.init(layers_optax)
        history = {"epoch": [], "train": [], "val": []}
        
        for epoch in range(1, n_epochs + 1):
            if report_progress:
                print(f"Epoch: {epoch}")

            layers_grad = self.backpropagation(inputs, targets)
            updates, opt_state = optimizer.update(layers_grad, opt_state, layers_optax)
            layers_optax = optax.apply_updates(layers_optax, updates)

            self.layers = layers_optax

            if record_every and epoch % record_every == 0:
                history["epoch"].append(epoch)
                history["train"].append(float(self.cost(inputs, targets)))
                history["val"].append(float(self.cost(X_val, y_val)) if X_val is not None else None)
        
        return history
          
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
                  batch_size = 10, lmbda = 0.0, penalty = "l2" ):
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
        self.penalty = penalty

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

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that results agree with numpy to ~1e-15
import jax.numpy as jnp                    # numpy-compatible functions that JAX can differentiate
import optax
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score,
                             classification_report,
                             roc_curve as sk_roc_curve, roc_auc_score, RocCurveDisplay)
from sklearn.base import BaseEstimator

# useful functions
def ReLU(z):
    return jnp.where(z > 0, z, 0.0)

def ReLU_der(z):
    return jnp.where(z > 0, 1.0, 0.0)

def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))

def mse(predict, target):
    return jnp.mean((predict - target) ** 2)

def mse_der(predict, target):
    return 2 * (predict - target) / predict.size

def cross_entropy(predict, target):
    return jnp.sum(-target * jnp.log(predict))

def binary_cross_entropy(predict, target):
    p = jnp.clip(predict, 1e-12, 1 - 1e-12)
    return -jnp.sum(target * jnp.log(p) + (1 - target) * jnp.log1p(-p))

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

def smce_der(predict, target):
    return predict - target 

# classification plotting:
def plot_classification(y, p, threshold = 0.5, label = "your classifier"):
      y_test, p_test = y, p          # replace by your test labels and output probabilities
      y_pred = p_test > threshold

      cm = confusion_matrix(y_test, y_pred)   # [[TN, FP], [FN, TP]]
      print(cm)
      print(f"accuracy {accuracy_score(y_test, y_pred):.4f}  precision {precision_score(y_test, y_pred, zero_division=0):.4f}  "
            f"recall {recall_score(y_test, y_pred):.4f}  F1 {f1_score(y_test, y_pred):.4f}")
      print(classification_report(y_test, y_pred, target_names=["negative", "positive"], zero_division=0))

      fpr, tpr, thresholds = sk_roc_curve(y_test, p_test, drop_intermediate=False)
      print(f"AUC {roc_auc_score(y_test, p_test):.4f}")

      fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
      ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"]).plot(ax=ax[0], colorbar=False)
      RocCurveDisplay.from_predictions(y_test, p_test, ax=ax[1], name = label)
      ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
      plt.tight_layout(); plt.show()

# NeuralNetwork class:

class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
    ):
        self.network_input_size = network_input_size    
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der

        vars = np.ones((len(layer_output_sizes), 2))
        self.initialize_layers(vars, rs = 2026)

    def __str__(self):
        return(f"Neural network with {len(self.layer_output_sizes) - 1} hidden layers.")

    def initialize_layers(self, vars, rs = 2026):
        """
        Make new layers with wieghts and biases from normal distribution with corresponding variances
        Note that length of vars must match number of layers. vars is indexed [(var_w1, var_b1), (var_w2, var_b2),...]
        """
        assert(len(vars) == len(self.layer_output_sizes)), "Number of variances must equal number of layers"
        rng = np.random.default_rng(rs)
        layers = []

        i_size = self.network_input_size
        for layer_output_size, sigmas in zip(self.layer_output_sizes, vars):
            sigma_w, sigma_b = sigmas
            W = rng.normal(loc = 0.0, scale = sigma_w, size =(i_size, layer_output_size))
            b = rng.normal(loc = 0.0, scale = sigma_b, size = layer_output_size)
            layers.append((W, b))

            i_size = layer_output_size

        self.layers = layers

    def predict(self, inputs):
        """
        Complete feed forward pass and prefict based on inputs
        """
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)

        return a

    def predict_layers(self, layers, inputs):
        """
        Complete feed forward pass and predict based on inputs with layers as input
        """
        a = inputs
        for (W, b), activation_func in zip(layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)

        return a

    def cost(self, inputs, targets):
        """
        Compute cost via the object's cost function on desired inputs and targets
        """
        a = self.predict(inputs)
        return self.cost_fun(a, targets)

    def cost_layers(self, inputs, layers, targets):
        """
        Compute cost via obect's cost function on desired inputs and targets with layers as input
        """
        a = self.predict_layers(layers, inputs)
        return self.cost_fun(a, targets)

    def _feed_forward_saver(self, inputs):
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

    def compute_gradient(self, inputs, targets):
        """
        Computes gradient via analytical backpropagation
        """

        layer_inputs, zs, predict = self._feed_forward_saver(inputs)

        layer_grads = [() for _ in self.layers]

        # We loop over the layers, from the last to the first
        for i in reversed(range(len(self.layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_ders[i]

            if i == len(self.layers) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = self.cost_der(predict, targets)
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W, b) = self.layers[i + 1]
                dC_da = dC_dz @ W.T

            dC_dz = dC_da * activation_der(z)
            dC_dW = layer_input.T @ dC_dz
            dC_db = np.sum(dC_dz, axis = 0)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def jax_gradient(self, inputs, layers,  targets):
        """
        Compute gradient with automatic differentiation via jax
        """
        grad = jax.grad(self.cost_layers,  argnums = 1)
        return grad(inputs, layers, targets)
    
    def update_weights(self, layer_grads, gamma):
        """
        Updates self.layers according to layer gradients. Equivalent to one step in
        plain gradient descent of learning rate gamma.
        """
        updated_layers = [(W - gamma * W_g, b - gamma * b_g) for (W, b), (W_g, b_g) in zip(self.layers, layer_grads)]
        self.layers = updated_layers

    def _make_batches(self, n, batch_size, rng):
        """
        Shuffle the indices and split them into minibatches. Private method used in sgd
        """
        idx = rng.permutation(n)
        return [idx[i:i + batch_size] for i in range(0, n, batch_size)]

    def train_sgd(
        self, inputs, targets, optimizer, rs = 2026, n_epochs = 50, batch_size = 5, autodiff = False, report_progress = False
        ):
        """
        Trains network using stochastic gradient descent.
        Pass an initialized optimizer as the optimizer argument.
        Gradually updates self.layers, returns None
        """

        layers_optax = self.layers
        opt_state = optimizer.init(layers_optax)

        rng = np.random.default_rng(rs)
        n_datapoints = inputs.shape[0]

        if autodiff == True:
            if report_progress == False:
                for _ in range(n_epochs):
                    for batch in self._make_batches(n_datapoints, batch_size, rng):
                        layers_grad = self.jax_gradient(inputs[batch], self.layers, targets[batch])
                        updates, opt_state = optimizer.update(layers_grad, opt_state, layers_optax)
                        layers_optax = optax.apply_updates(layers_optax, updates)

                        self.layers = layers_optax
            else:
                for _ in range(n_epochs ):
                    print(f"Epoch: {_ + 1}")
                    for batch in self._make_batches(n_datapoints, batch_size, rng):
                        layers_grad = self.jax_gradient(inputs[batch], self.layers, targets[batch])
                        updates, opt_state = optimizer.update(layers_grad, opt_state, layers_optax)
                        layers_optax = optax.apply_updates(layers_optax, updates)

                        self.layers = layers_optax
        else:
            if report_progress == False:
                for _ in range(n_epochs):
                    for batch in self._make_batches(n_datapoints, batch_size, rng):
                        layers_grad = self.compute_gradient(inputs[batch], targets[batch])
                        updates, opt_state = optimizer.update(layers_grad, opt_state, layers_optax)
                        layers_optax = optax.apply_updates(layers_optax, updates)

                        self.layers = layers_optax
            else:
                for _ in range(n_epochs ):
                    print(f"Epoch: {_ + 1}")
                    for batch in self._make_batches(n_datapoints, batch_size, rng):
                        layers_grad = self.compute_gradient(inputs[batch], targets[batch])
                        updates, opt_state = optimizer.update(layers_grad, opt_state, layers_optax)
                        layers_optax = optax.apply_updates(layers_optax, updates)

                        self.layers = layers_optax




    def train_pgd(
            self, inputs, targets, optimizer, n_epochs = 100, autodiff = False, report_progress = False
    ):
        """
        Trains network plain gradient descent.
        Pass an initialized optax optimizer as the optimizer argument
        Gradually updatater self.layers, returns None
        """

        layers_optax = self.layers
        opt_state = optimizer.init(layers_optax)


        if autodiff == True:
            if report_progress == False:
                for _ in range(n_epochs):
                    layers_grad = self.jax_gradient(inputs, self.layers, targets)
                    updates, opt_state = optimizer.update(layers_grad, opt_state, layers_optax)
                    layers_optax = optax.apply_updates(layers_optax, updates)

                    self.layers = layers_optax
            else:
                for _ in range(n_epochs):
                    print(f"Epoch: {_ + 1}")
                    layers_grad = self.jax_gradient(inputs, self.layers, targets)
                    updates, opt_state = optimizer.update(layers_grad, opt_state, layers_optax)
                    layers_optax = optax.apply_updates(layers_optax, updates)

                    self.layers = layers_optax
        else:
            if report_progress == False:
                for _ in range(n_epochs):
                    layers_grad = self.compute_gradient(inputs, targets)
                    updates, opt_state = optimizer.update(layers_grad, opt_state, layers_optax)
                    layers_optax = optax.apply_updates(layers_optax, updates)

                    self.layers = layers_optax
            else:
                for _ in range(n_epochs):
                    print(f"Epoch: {_ + 1}")
                    layers_grad = self.compute_gradient(inputs, self.layers, targets)
                    updates, opt_state = optimizer.update(layers_grad, opt_state, layers_optax)
                    layers_optax = optax.apply_updates(layers_optax, updates)            
        
    def save_layers(self):
        """
        Method used to save layers for later validation so layers can be altered
        """
        return self.layers




class Estimator(NeuralNetwork, BaseEstimator):

    def __init__(
        self,
        layer_output_sizes = (50, 1),
        activation_funcs = (ReLU, sigmoid),
        activation_ders = (ReLU_der, lambda _: 1),
        cost_fun = binary_cross_entropy,
        cost_der = smce_der,
        random_seed = 2026,
        gamma = 0.03,
        epochs = 1000,
        batch_size = 10,
        lmbda = 0.0
    ):    
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.random_seed = random_seed
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.lmbda = lmbda


    def initialize_layers_(self):
        """
        Make new layers with wieghts and biases from normal distribution with corresponding variances
        Note that length of vars must match number of layers. vars is indexed [(var_w1, var_b1), (var_w2, var_b2),...]
        """

        vars = np.zeros_like(self.layer_output_sizes)
        vars[0] = self.input_size
        vars[1:] = self.layer_output_sizes[:-1]
        rng = np.random.default_rng(self.random_seed)
        layers = []

        i_size = self.input_size
        for layer_output_size, sigma_w in zip(self.layer_output_sizes, vars):
            W = rng.normal(loc = 0.0, scale = sigma_w, size =(i_size, layer_output_size))
            b = np.zeros(layer_output_size)
            layers.append((W, b))

            i_size = layer_output_size

        self.layers = layers

    def fit(self, X, y):

        self.input_size = X.shape[1]
        self.initialize_layers_()
        optimizer = optax.adam(learning_rate = self.gamma)
        self.train_sgd(X, y, optimizer, self.random_seed, self.epochs, self.batch_size)
        self.layers_ = self.layers
        return self

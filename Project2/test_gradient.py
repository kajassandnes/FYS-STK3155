"""
Check NeuralNetwork gradients.

JAX automatic differentiation is used only here, not in training.
Backpropagation is compared to jax.grad of the same cost, including
the L1 / L2 penalty when lambda > 0. A linear network is also trained
with plain GD and checked against closed-form OLS.

Use of LLM:
The tests in this file is written by a LLM
"""
import numpy as np
import jax.numpy as jnp
import optax
from neural_network import NeuralNetwork
from utils import (
    mse, mse_der,
    sigmoid, sigmoid_der,
    ReLU, ReLU_der,
    identity, identity_der,
)
from runge_data import runge_data, design_matrix, split_scale, OLS

SEED = 2026
ATOL = 1e-12


def _as_numpy(grad):
    """Convert a list of (W, b) jax arrays to numpy."""
    return [(np.asarray(W), np.asarray(b)) for W, b in grad]


def _max_abs_diff(grad_bp, grad_jax):
    """Largest |backprop - jax| over all layers, weights and biases."""
    diffs = []
    for (W_bp, b_bp), (W_jax, b_jax) in zip(grad_bp, grad_jax):
        diffs.append(np.max(np.abs(np.asarray(W_bp) - W_jax)))
        diffs.append(np.max(np.abs(np.asarray(b_bp) - b_jax)))
    return max(diffs)


def assert_grads_match(net, X, y, name):
    """Compare backpropagation to jax.grad on every W and b."""
    X_j = jnp.asarray(X)
    y_j = jnp.asarray(y)
    layers_j = [(jnp.asarray(W), jnp.asarray(b)) for W, b in net.layers]

    grad_bp = net.backpropagation(X, y)
    grad_jax = _as_numpy(net.jax_gradient(X_j, layers_j, y_j))

    err = _max_abs_diff(grad_bp, grad_jax)
    assert err < ATOL, f"{name}: max |backprop - jax| = {err:.3e} (tol {ATOL:.0e})"
    print(f"{name:32}  max |Δ| = {err:.3e}  OK")


def make_data(n_samples, n_in, n_out, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_in))
    y = rng.normal(size=(n_samples, n_out))
    return X, y


def make_net(n_in, sizes, acts, ders, lmbda=0.0, penalty="l2", init=None, seed=SEED):
    return NeuralNetwork(
        n_in, sizes, acts, ders, mse, mse_der,
        lmbda=lmbda, seed=seed, init=init, penalty=penalty,
    )


# ---------------------------------------------------------------------------
# Backprop vs JAX
# ---------------------------------------------------------------------------

def test_linear_identity():
    """No hidden layer, identity output: y = X W + b."""
    n_in, n_out = 3, 1
    X, y = make_data(8, n_in, n_out)
    net = make_net(n_in, [n_out], [identity], [identity_der])
    assert_grads_match(net, X, y, "linear, identity")


def test_one_hidden_sigmoid():
    X, y = make_data(6, n_in=2, n_out=1)
    net = make_net(
        2, [4, 1],
        [sigmoid, identity], [sigmoid_der, identity_der],
    )
    assert_grads_match(net, X, y, "1 hidden, sigmoid")


def test_two_hidden_sigmoid():
    X, y = make_data(6, n_in=2, n_out=1)
    net = make_net(
        2, [5, 3, 1],
        [sigmoid, sigmoid, identity],
        [sigmoid_der, sigmoid_der, identity_der],
    )
    assert_grads_match(net, X, y, "2 hidden, sigmoid")


def test_one_hidden_relu():
    X, y = make_data(6, n_in=2, n_out=1)
    net = make_net(
        2, [4, 1],
        [ReLU, identity], [ReLU_der, identity_der],
        init="He",
    )
    assert_grads_match(net, X, y, "1 hidden, ReLU")


def test_two_hidden_relu():
    X, y = make_data(6, n_in=2, n_out=1)
    net = make_net(
        2, [5, 3, 1],
        [ReLU, ReLU, identity],
        [ReLU_der, ReLU_der, identity_der],
        init="He",
    )
    assert_grads_match(net, X, y, "2 hidden, ReLU")


def test_l2_penalty():
    X, y = make_data(6, n_in=2, n_out=1)
    net = make_net(
        2, [4, 1],
        [sigmoid, identity], [sigmoid_der, identity_der],
        lmbda=0.1, penalty="l2",
    )
    assert_grads_match(net, X, y, "L2, lambda=0.1")


def test_l1_penalty():
    X, y = make_data(6, n_in=2, n_out=1)
    net = make_net(
        2, [4, 1],
        [ReLU, identity], [ReLU_der, identity_der],
        lmbda=0.1, penalty="l1", init="He",
    )
    assert_grads_match(net, X, y, "L1, lambda=0.1")


# ---------------------------------------------------------------------------
# OLS recovery: linear net + plain GD vs closed form
# ---------------------------------------------------------------------------

def test_ols_recovery(degree=2, n_epochs=5000):
    """
    A network with no hidden layer and identity output is linear regression.
    Plain GD must recover the OLS coefficients (including the intercept = bias).
    """
    x, y = runge_data(rs=SEED)
    y = y.reshape(-1, 1)
    X = design_matrix(x, degree)
    X_tr, _, y_tr, _ = split_scale(X, y, ts=0.2, rs=SEED)

    X_aug = np.column_stack([np.ones(len(y_tr)), X_tr])
    theta = OLS(X_aug, y_tr).ravel()

    n = len(y_tr)
    hess = (2.0 / n) * (X_aug.T @ X_aug)
    eta_max = 2.0 / np.linalg.eigvalsh(hess).max()
    eta = 0.5 * float(eta_max)

    net = make_net(X_tr.shape[1], [1], [identity], [identity_der], lmbda=0.0)
    net.train_pgd(X_tr, y_tr, optax.sgd(eta), n_epochs=n_epochs)

    W, b = net.layers[0]
    W, b = np.asarray(W).ravel(), np.asarray(b).ravel()
    err = max(np.max(np.abs(W - theta[1:])), np.max(np.abs(b - theta[:1])))
    assert err < 1e-6, f"OLS recovery: max |W,b - theta| = {err:.3e}"
    print(f"{'OLS recovery, degree ' + str(degree):32}  max |Δ| = {err:.3e}  OK")
    print(f"    eta = {eta:.4f}  (eta_max = {eta_max:.4f})")


def run_all():
    print("Backprop vs JAX")
    test_linear_identity()
    test_one_hidden_sigmoid()
    test_two_hidden_sigmoid()
    test_one_hidden_relu()
    test_two_hidden_relu()
    test_l2_penalty()
    test_l1_penalty()
    print("\nOLS recovery")
    test_ols_recovery()
    print("\nAll tests passed.")


if __name__ == "__main__":
    run_all()
